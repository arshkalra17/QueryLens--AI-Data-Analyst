"""Data-quality checks producing a scored report with recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import is_string_series


@dataclass
class QualityIssue:
    column: str | None
    kind: str
    message: str
    severity: str  # high / medium / low

    def to_dict(self) -> dict:
        return {"column": self.column, "kind": self.kind, "message": self.message, "severity": self.severity}


@dataclass
class QualityReport:
    score: int
    issues: list[QualityIssue] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "checks": self.checks,
        }


def quality_report(df: pd.DataFrame) -> QualityReport:
    """Evaluate data quality, returning a 0-100 score plus detailed issues."""
    issues: list[QualityIssue] = []
    checks: dict = {}
    n_rows = len(df)
    if n_rows == 0:
        return QualityReport(score=0, issues=[QualityIssue(None, "empty", "Dataset is empty", "high")], checks={"rows": 0})

    total_penalty = 0.0

    # 1. Duplicates
    dup = int(df.duplicated().sum())
    checks["duplicate_rows"] = dup
    if dup:
        severity = "high" if dup / n_rows > 0.05 else "medium"
        issues.append(QualityIssue(None, "duplicates", f"{dup:,} duplicate rows ({dup/n_rows*100:.1f}%).", severity))
        total_penalty += min(25, dup / n_rows * 100)

    # 2. Missing values
    missing = df.isna().sum()
    missing_total = int(missing.sum())
    checks["missing_values"] = missing_total
    if missing_total:
        missing_cols = missing[missing > 0].sort_values(ascending=False)
        worst = missing_cols.index[0]
        share = missing_cols.iloc[0] / n_rows * 100
        severity = "high" if share > 30 else ("medium" if share > 10 else "low")
        issues.append(
            QualityIssue(
                str(worst), "missing",
                f"'{worst}' has {int(missing_cols.iloc[0]):,} missing values ({share:.1f}%).",
                severity,
            )
        )
        total_penalty += min(30, missing_total / (n_rows * len(df.columns)) * 100)

    # 3. Constant columns
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            issues.append(QualityIssue(str(col), "constant", f"'{col}' is constant and adds no information.", "low"))
            total_penalty += 3

    # 4. Type issues in numeric columns
    for col in [str(c) for c in df.columns if is_string_series(df[c])][:8]:
        s = df[col].dropna().astype(str).str.replace(",", "").str.strip()
        numeric_mask = s.str.match(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
        if len(s):
            non_numeric = int((~numeric_mask).sum())
            if 0 < non_numeric <= len(s) * 0.5:
                issues.append(
                    QualityIssue(str(col), "type",
                                 f"'{col}' looks numeric but {non_numeric:,} of {len(s):,} values are non-numeric text.", "medium")
                )
                total_penalty += min(10, non_numeric / len(s) * 20)

    # 5. Whitespace / empty-string hygiene
    for col in [str(c) for c in df.columns if is_string_series(df[c])][:8]:
        s = df[col].dropna().astype(str)
        if len(s):
            blank = int(((s == "") | (s.str.strip() == "") | (s.str.lower() == "nan")).sum())
            if blank:
                issues.append(QualityIssue(str(col), "hygiene", f"'{col}' has {blank:,} empty/blank string values.", "low"))
                total_penalty += min(8, blank / len(s) * 15)

    # 6. Outliers (numeric)
    outlier_count = 0
    for col in df.select_dtypes(include=np.number).columns[:8]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr and np.isfinite(iqr):
            outlier_count += int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    checks["outliers"] = outlier_count
    if outlier_count:
        issues.append(QualityIssue(None, "outliers", f"Approx {outlier_count:,} numeric outliers found across columns.", "low"))
        total_penalty += min(8, outlier_count / max(1, n_rows) * 5)

    # 7. Date consistency
    for col in df.columns:
        if is_string_series(df[col]) and df[col].nunique(dropna=True) > 10:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.9:
                    invalid = int(parsed.isna().sum())
                    if invalid:
                        issues.append(QualityIssue(str(col), "date", f"'{col}' has {invalid:,} unparsable dates.", "medium"))
                        total_penalty += min(10, invalid / len(df) * 10)
                    break
            except Exception:
                continue

    checks["columns"] = int(len(df.columns))
    checks["rows"] = int(n_rows)
    score = max(0, min(100, int(round(100 - total_penalty))))
    issues.sort(key=lambda i: {"high": 0, "medium": 1, "low": 2}[i.severity])
    return QualityReport(score=score, issues=issues, checks=checks)
