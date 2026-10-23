"""In-memory dataset registry backed by the filesystem for persistence.

Each uploaded CSV is copied to ``data/uploads`` and re-loaded automatically at
startup so datasets survive restarts.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_settings
from ..utils import is_string_series, to_json_safe
from .logging_setup import metrics


def _serialize(value: Any) -> Any:
    """Convert a scalar into a JSON friendly value."""
    return to_json_safe(value)


@dataclass
class ColumnInfo:
    name: str
    dtype: str
    numeric: bool
    categorical: bool
    datetime: bool
    nullable: bool
    null_count: int
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetRecord:
    dataset_id: str
    name: str
    source: str
    uploaded_at: float
    row_count: int
    col_count: int
    columns: list[ColumnInfo]
    preview: list[dict]
    file_size: int
    sha256: str
    df: pd.DataFrame = field(repr=False, compare=False)
    full: list[dict] = field(default_factory=list, repr=False, compare=False)

    def schema_snippet(self, max_rows: int = 5, max_cols: int = 12) -> str:
        """Compact textual schema used in prompts and reasoning."""
        safe = "".join(c if c.isalnum() else "_" for c in self.name.lower())[:60]
        lines = [
            f"dataset: {self.name} (id={self.dataset_id}, table={safe}, {self.row_count} rows, {self.col_count} cols)"
        ]
        for c in self.columns[:max_cols]:
            kind = []
            if c.numeric:
                kind.append("numeric")
            if c.categorical:
                kind.append("categorical")
            if c.datetime:
                kind.append("datetime")
            lines.append(
                f"  - {c.name} [{c.dtype}] ({','.join(kind)}): {c.null_count} nulls, "
                f"{c.unique_count} unique, e.g. {[str(v)[:40] for v in c.sample_values[:5]]}"
            )
        if self.col_count > max_cols:
            lines.append(f"  ... and {self.col_count - max_cols} more columns")
        return "\n".join(lines)

    def to_meta(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "uploaded_at": self.uploaded_at,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "file_size": self.file_size,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "numeric": c.numeric,
                    "categorical": c.categorical,
                    "datetime": c.datetime,
                    "null_count": c.null_count,
                    "unique_count": c.unique_count,
                    "sample_values": c.sample_values[:5],
                }
                for c in self.columns
            ],
        }


def classify_columns(df: pd.DataFrame) -> list[ColumnInfo]:
    cols: list[ColumnInfo] = []
    for name in df.columns:
        series = df[name]
        dtype = str(series.dtype)
        nulls = int(series.isna().sum())
        uniques = int(series.nunique(dropna=True))
        numeric = pd.api.types.is_numeric_dtype(series)
        datetime = pd.api.types.is_datetime64_any_dtype(series)
        categorical = is_string_series(series) or (numeric and uniques <= 40)
        samples = [_serialize(v) for v in series.dropna().head(8).tolist()]
        cols.append(
            ColumnInfo(
                name=str(name),
                dtype=dtype,
                numeric=numeric,
                categorical=categorical,
                datetime=datetime,
                nullable=nulls > 0,
                null_count=nulls,
                unique_count=uniques,
                sample_values=samples,
            )
        )
    return cols


def make_preview(df: pd.DataFrame, n: int = 10) -> list[dict]:
    return [_serialize_dict(row) for row in df.head(n).to_dict(orient="records")]


def _serialize_dict(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        v = _serialize(v)
        out[str(k)] = None if isinstance(v, float) and v != v else v
    return out


def _persist_stem(name: str) -> str:
    """Filesystem-safe stem for a dataset name, WITHOUT double-adding the suffix.

    The old code wrote ``f"{name}.csv"`` where ``name`` already kept its original
    extension, so "customers-1000 2.csv" persisted as "customers-1000 2_csv.csv"
    and reloaded with the mangled name "..._csv.csv" -> table "customers_1000_2_csv_csv".
    Strip a known extension first so the persisted file keeps the original name.
    """
    base = name
    for ext in (".csv", ".tsv", ".txt"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in base)[:60]


class DataFrameRegistry:
    def __init__(self, upload_dir: Path | None = None) -> None:
        settings = get_settings()
        self._upload_dir = upload_dir or settings.upload_dir_resolved
        self._datasets: dict[str, DatasetRecord] = {}
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def startup(self) -> int:
        """Load every persisted file in the upload dir. Returns count loaded."""
        if not self._upload_dir.exists():
            return 0
        loaded = 0
        for path in sorted(self._upload_dir.glob("*.csv")):
            try:
                record = self._from_file(path, path.name)
                if record is not None:
                    self._datasets[record.dataset_id] = record
                    loaded += 1
            except Exception as exc:  # pragma: no cover - defensive
                import logging
                logging.getLogger(__name__).warning("Failed to reload %s: %s", path, exc)
        return loaded

    # -- core operations ---------------------------------------------------
    def register(self, df: pd.DataFrame, name: str, source: str) -> DatasetRecord:
        with self._lock:
            if len(self._datasets) >= get_settings().max_datasets:
                raise ValueError(f"Dataset limit reached ({get_settings().max_datasets}). Delete a dataset first.")
            dataset_id = uuid.uuid4().hex[:12]
            df = df.copy()
            df.columns = [str(c).strip() for c in df.columns]
            if df.columns.duplicated().any():
                raise ValueError("Duplicate column names found. Rename columns before uploading.")
            # best-effort datetime coercion for common date-ish columns
            df = _coerce_datetimes(df)

            record = DatasetRecord(
                dataset_id=dataset_id,
                name=name,
                source=source,
                uploaded_at=time.time(),
                row_count=int(len(df)),
                col_count=int(len(df.columns)),
                columns=classify_columns(df),
                preview=make_preview(df),
                file_size=0,
                sha256=_df_hash(df),
                df=df,
                full=_full(df),
            )
            self._datasets[dataset_id] = record
            metrics.inc("datasets_registered")
            return record

    def register_from_path(self, path: Path) -> DatasetRecord:
        record = self._from_file(path, path.name)
        if record is None:
            raise ValueError("File could not be loaded as CSV")
        with self._lock:
            self._datasets[record.dataset_id] = record
        return record

    def persist(self, record: DatasetRecord) -> Path:
        """Write a dataset back to disk (idempotent, keyed by name)."""
        safe = _persist_stem(record.name)
        path = self._upload_dir / f"{safe}.csv"
        record.df.to_csv(path, index=False)
        record.file_size = path.stat().st_size
        return path

    def get(self, dataset_id: str) -> DatasetRecord | None:
        with self._lock:
            return self._datasets.get(dataset_id)

    def require(self, dataset_id: str) -> DatasetRecord:
        record = self.get(dataset_id)
        if record is None:
            raise KeyError(f"Unknown dataset id: {dataset_id}")
        return record

    def list(self) -> list[DatasetRecord]:
        with self._lock:
            return list(self._datasets.values())

    def delete(self, dataset_id: str) -> bool:
        with self._lock:
            record = self._datasets.pop(dataset_id, None)
        if record is None:
            return False
        # delete matching persisted file
        safe = _persist_stem(record.name)
        path = self._upload_dir / f"{safe}.csv"
        if path.exists():
            path.unlink()
        return True

    def search(self, term: str) -> list[dict]:
        """Keyword match across dataset names / column names."""
        term = term.lower()
        out = []
        for r in self.list():
            score = 0
            if term in r.name.lower():
                score += 10
            for c in r.columns:
                if term in c.name.lower():
                    score += 3
            if score:
                out.append({"dataset_id": r.dataset_id, "name": r.name, "score": score, "columns": [c.name for c in r.columns]})
        return sorted(out, key=lambda x: -x["score"])

    # -- internals ---------------------------------------------------------
    def _from_file(self, path: Path, name: str) -> DatasetRecord | None:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty or len(df.columns) == 0:
            return None
        record = self.register(df, name, str(path))
        record.file_size = path.stat().st_size
        return record


def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Try to parse string columns that look like dates."""
    for col in df.columns:
        s = df[col]
        if not is_string_series(s):
            continue
        lowered = s.dropna().astype(str).str.lower()
        if lowered.empty:
            continue
        first = lowered.iloc[0]
        looks_like_date = (
            "/" in first or "-" in first and any(c.isdigit() for c in first) and len(first) >= 8
        ) and not lowered.str.isnumeric().all()
        if looks_like_date:
            try:
                parsed = pd.to_datetime(s, errors="raise")
                if parsed.notna().mean() > 0.9:
                    df[col] = parsed
            except Exception:
                pass
    return df


def _df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.head(200).to_csv(index=False).encode()).hexdigest()[:16]


def _full(df: pd.DataFrame, limit: int = 200) -> list[dict]:
    """Cap the cached full view so we don't hold huge duplicated data in memory."""
    return [_serialize_dict(row) for row in df.head(limit).to_dict(orient="records")]


_registry: DataFrameRegistry | None = None


def get_registry() -> DataFrameRegistry:
    global _registry
    if _registry is None:
        _registry = DataFrameRegistry()
    return _registry


def reset_registry() -> None:
    global _registry
    _registry = None
