"""Lightweight, dependency-free forecasting.

Decompose a time series into a linear trend plus a seasonal component
(day-of-week or month-of-year). Project `periods` ahead and attach a
confidence band derived from the residual standard deviation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    periods: int = 12,
    seasonality: str = "month",
    confidence: float = 0.95,
) -> dict:
    """Return a dict with forecast table + model summary.

    ``seasonality``: "month", "dayofweek", or "none".
    """
    if periods <= 0:
        raise ValueError("periods must be > 0")
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[date_column, value_column]).sort_values(date_column)

    if len(work) < 4:
        raise ValueError("Not enough data points to forecast (need >= 4 valid rows)")

    # aggregate to the seasonality period to stabilise the series
    freq = {"month": "ME", "dayofweek": "D", "none": "ME"}.get(seasonality, "ME")
    period_series = work.set_index(date_column)[value_column].resample(freq).mean().dropna()
    if len(period_series) < 3:
        raise ValueError("Not enough periods after aggregation to forecast")

    values = period_series.values.astype(float)
    n = len(values)

    # linear trend
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    trend = intercept + slope * x
    detrended = values - trend

    # seasonal component
    seasonal_index = _seasonal_positions(period_series.index, seasonality)
    seasonal = np.zeros(n)
    for i in range(n):
        mask = seasonal_index == seasonal_index[i]
        seasonal[i] = detrended[mask].mean()

    resid = values - trend - seasonal
    resid_std = resid.std(ddof=1) if len(resid) > 2 else 0.0
    z = 1.96 if confidence >= 0.99 else (1.645 if confidence >= 0.95 else 1.28)

    # extend into future
    future_dates = _future_periods(period_series.index, periods, seasonality)
    future_x = np.arange(n, n + periods, dtype=float)
    future_trend = intercept + slope * future_x
    future_seasonal = np.zeros(periods)
    f_idx = _seasonal_positions(future_dates, seasonality)
    for i in range(periods):
        mask = seasonal_index == f_idx[i]
        future_seasonal[i] = detrended[mask].mean() if mask.any() else 0.0
    future = future_trend + future_seasonal

    forecast_series = pd.Series(future, index=future_dates, name=value_column)

    # quality metrics
    sse = float((resid ** 2).sum())
    mae = float(np.abs(resid).mean())
    rmse = float(np.sqrt(sse / max(1, len(resid))))
    mean_abs = float(np.abs(values).mean()) or 1.0
    mape = min(999.0, mae / mean_abs * 100)

    table = []
    all_dates = list(period_series.index) + list(future_dates)
    for i, d in enumerate(all_dates):
        if i < n:
            table.append(
                {"date": pd.Timestamp(d).strftime("%Y-%m-%d"), "type": "actual", "value": round(float(values[i]), 3)}
            )
        else:
            j = i - n
            table.append(
                {
                    "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "type": "forecast",
                    "value": round(float(future[j]), 3),
                    "lower": round(float(future[j] - z * resid_std), 3),
                    "upper": round(float(future[j] + z * resid_std), 3),
                }
            )

    direction = "upward" if slope >= 0 else "downward"
    last = values[-1]
    last_fc = future[-1]
    change = (last_fc - last) / abs(last) * 100 if last else 0.0

    return {
        "date_column": date_column,
        "value_column": value_column,
        "periods": periods,
        "seasonality": seasonality,
        "model": {
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "direction": direction,
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "mape_pct": round(mape, 2),
            "residual_std": round(float(resid_std), 3),
            "confidence": confidence,
        },
        "summary": (
            f"The series is trending {direction} by {abs(slope):,.3f} units per period "
            f"({slope/abs(last)*100 if last else 0:+.1f}% of the last actual). "
            f"Next {periods} period(s) are projected to end at {future[-1]:,.1f}, "
            f"a {change:+.1f}% change from the last actual value."
        ),
        "table": table,
    }


def _seasonal_positions(index: pd.DatetimeIndex, seasonality: str) -> np.ndarray:
    if seasonality == "dayofweek":
        return index.dayofweek.values
    if seasonality == "month":
        return index.month.values
    return np.zeros(len(index), dtype=int)


def _future_periods(index: pd.DatetimeIndex, periods: int, seasonality: str) -> pd.DatetimeIndex:
    if seasonality == "dayofweek":
        last = index[-1]
        out = []
        d = last
        while len(out) < periods:
            d = d + pd.Timedelta(days=1)
            out.append(d)
        return pd.DatetimeIndex(out)
    last = index[-1]
    per = index.to_period("M")[-1]
    out = []
    for k in range(1, periods + 1):
        out.append((per + k).to_timestamp())
    return pd.DatetimeIndex(out)
