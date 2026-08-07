"""Dataset upload / validation / listing endpoints."""
from __future__ import annotations

import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from ...core.registry import DataFrameRegistry, get_registry
from ...core.sessions import get_sessions
from ..schemas import DeleteResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _registry() -> DataFrameRegistry:
    return get_registry()


def _validate_csv(content: bytes, filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"'{filename}' is not a valid CSV: {exc}")
    if df.empty:
        raise HTTPException(status_code=422, detail=f"'{filename}' has no rows")
    if df.columns.duplicated().any():
        raise HTTPException(status_code=422, detail=f"'{filename}' has duplicate column names")
    if df.shape[1] == 1 and df.iloc[:, 0].nunique(dropna=True) == 1:
        raise HTTPException(status_code=422, detail=f"'{filename}' looks malformed (single column)")
    return df


@router.post("")
def upload(
    files: list[UploadFile] = File(...),
    session_id: str | None = None,
    registry: DataFrameRegistry = Depends(_registry),
):
    """Upload one or more CSV files. Each file is validated and registered."""
    imported = []
    errors = []
    for upload in files:
        raw = upload.file.read()
        if len(raw) > get_max_upload_bytes():
            errors.append({"file": upload.filename, "error": "file exceeds size limit"})
            continue
        try:
            df = _validate_csv(raw, upload.filename or "data.csv")
        except HTTPException as exc:
            errors.append({"file": upload.filename, "error": str(exc.detail)})
            continue
        name = (upload.filename or "data.csv").rsplit("/", 1)[-1]
        # Deduplicate: re-uploading a file with the same name replaces the old
        # dataset (removes its RAG index + persisted file) instead of stacking
        # duplicate entries. The fresh upload is pinned as the active dataset.
        for existing in registry.list():
            if existing.name == name:
                registry.delete(existing.dataset_id)
                _drop_index(existing.dataset_id)
                break
        try:
            record = registry.register(df, name, f"upload:{name}")
            registry.persist(record)
        except ValueError as exc:
            errors.append({"file": upload.filename, "error": str(exc)})
            continue
        _index_record(record)
        if session_id:
            try:
                get_sessions().attach_dataset(session_id, record.dataset_id)
            except KeyError:
                pass
        imported.append(record.to_meta())

    status_code = 207 if errors and imported else (400 if errors else 200)
    payload = {"imported": imported, "errors": errors, "count": len(imported)}
    from fastapi.responses import JSONResponse

    return JSONResponse(payload, status_code=status_code)


@router.get("")
def list_datasets(registry: DataFrameRegistry = Depends(_registry)):
    return {"datasets": [r.to_meta() for r in registry.list()], "count": len(registry.list())}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, registry: DataFrameRegistry = Depends(_registry)):
    record = registry.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")
    return record.to_meta()


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, rows: int = 10, registry: DataFrameRegistry = Depends(_registry)):
    record = registry.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")
    return {"dataset_id": dataset_id, "name": record.name, "rows": record.preview[:rows], "total_rows": record.row_count}


@router.get("/{dataset_id}/export")
def export_csv(dataset_id: str, registry: DataFrameRegistry = Depends(_registry)):
    record = registry.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")
    from fastapi.responses import Response

    csv_bytes = record.df.to_csv(index=False).encode()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{record.name}.csv"'},
    )


@router.delete("/{dataset_id}", response_model=DeleteResponse)
def delete_dataset(dataset_id: str, registry: DataFrameRegistry = Depends(_registry)):
    deleted = registry.delete(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Unknown dataset id: {dataset_id}")
    _drop_index(dataset_id)
    return {"deleted": True, "dataset_id": dataset_id}


def _index_record(record) -> None:
    from ...rag import get_rag

    try:
        get_rag().index_record(record)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("RAG index failed for %s: %s", record.name, exc)


def _drop_index(dataset_id: str) -> None:
    from ...rag import get_rag

    try:
        get_rag().drop_record(dataset_id)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("RAG drop failed for %s: %s", dataset_id, exc)


def get_max_upload_bytes() -> int:
    from ...config import get_settings

    return get_settings().max_upload_mb * 1024 * 1024
