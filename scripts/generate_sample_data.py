"""Generate realistic sample CSVs with known quirks (outliers, trends,
seasonality, missing values) so every feature can be demoed and tested.

Usage: python scripts/generate_sample_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
rng = np.random.default_rng(42)


def sales_data(rows: int = 2400) -> pd.DataFrame:
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse", "Headset", "Charger"]
    channels = ["Online", "Retail", "Partner", "Distributor"]
    sales_reps = ["Aarav", "Bianca", "Chen", "Diana", "Eli", "Fatima", "Gaurav", "Hana"]

    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    region = rng.choice(regions, rows, p=[0.24, 0.22, 0.20, 0.19, 0.15])
    product = rng.choice(products, rows)
    base_price = pd.Series(rng.uniform(20, 1500, rows))
    units = np.clip(np.random.lognormal(2.2, 0.8, rows).astype(int), 1, 500)

    # seasonality: December + small weekly bump
    seasonal = (1 + 0.25 * ((dates.month == 12)) + 0.05 * ((dates.dayofweek < 5)))
    region_factor = pd.Series({"North": 1.0, "South": 0.9, "East": 1.05, "West": 1.15, "Central": 0.8}).reindex(region).values

    revenue = base_price * units * seasonal * region_factor

    # inject anomalies: a few huge/spike orders
    spike_idx = rng.choice(rows, 12, replace=False)
    revenue.iloc[spike_idx] *= rng.uniform(4, 9, len(spike_idx))

    # missing values sprinkled in
    df = pd.DataFrame(
        {
            "date": dates,
            "region": region,
            "product_category": product,
            "units_sold": units,
            "unit_price": base_price.round(2),
            "revenue": revenue.round(2),
            "cost": (revenue * rng.uniform(0.35, 0.7, rows)).round(2),
            "channel": rng.choice(channels, rows),
            "customer_id": rng.choice([f"CUS{i:04d}" for i in range(1, 121)], rows),
            "sales_rep": rng.choice(sales_reps, rows),
        }
    )
    mask = rng.random(rows) < 0.015
    df.loc[mask, "sales_rep"] = None
    return df


def retail_transactions(rows: int = 1500) -> pd.DataFrame:
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata"]
    items = ["T-Shirt", "Jeans", "Shoes", "Jacket", "Hat", "Socks", "Belt", "Scarf"]
    payment = ["UPI", "Card", "Cash", "NetBanking", "Wallet"]

    dates = pd.date_range("2024-01-01", periods=rows, freq="h").strftime("%Y-%m-%d %H:%M")
    city = rng.choice(cities, rows)
    item = rng.choice(items, rows)
    price = pd.Series(rng.uniform(199, 4999, rows))
    qty = rng.integers(1, 5, rows)
    total = (price * qty).round(2)
    # top customers get more business
    customer_pool = [f"CUST{i:05d}" for i in range(1, 61)] + [f"CUST{i:05d}" for i in range(90, 101)]
    p = np.ones(len(customer_pool))
    p[:40] = 2.2
    p[90:] = 6.0
    p = p / p.sum()
    customer = rng.choice(customer_pool, rows, p=p)

    rating = rng.integers(1, 6, rows)
    # inject an anomaly: one city with a few huge basket totals
    anomaly = np.zeros(rows, dtype=bool)
    idx = rng.choice(rows, 8, replace=False)
    anomaly[idx] = True
    total.iloc[idx] *= rng.uniform(8, 15, len(idx))

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN{i:06d}" for i in range(1, rows + 1)],
            "date": dates,
            "customer_id": customer,
            "customer_name": [f"Customer {c}" for c in customer],
            "city": city,
            "item": item,
            "quantity": qty,
            "unit_price": price.round(2),
            "total": total,
            "payment_method": rng.choice(payment, rows),
            "rating": rating,
        }
    )
    df.loc[rng.random(rows) < 0.02, "city"] = None
    return df


def web_traffic(rows: int = 730) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    trend = np.linspace(0, 1, rows) * 400
    weekly = np.where(dates.dayofweek >= 5, -120, 60)
    seasonal = 180 * np.sin(np.linspace(0, 8 * np.pi, rows))
    noise = rng.normal(0, 180, rows)
    pageviews = trend + weekly + seasonal + noise
    pageviews = np.clip(pageviews, 200, None)

    conversion = 3.2 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, rows)) + rng.normal(0, 0.3, rows)

    # inject: a massive spike (viral) and a crash (outage) — anomalies
    spike_at = rng.integers(50, rows - 40)
    pageviews[spike_at] *= 2.6
    pageviews[spike_at + 3] *= 0.35

    ad_spend = 800 + 200 * np.sin(np.linspace(0, 6 * np.pi, rows)) + rng.normal(0, 90, rows)
    bounce = 55 - 8 * np.sin(np.linspace(0, 4 * np.pi, rows)) + rng.normal(0, 4, rows)

    return pd.DataFrame(
        {
            "date": dates,
            "pageviews": pageviews.round(0).astype(int),
            "sessions": (pageviews * 0.62).round(0).astype(int),
            "conversion_rate": conversion.round(2),
            "bounce_rate": bounce.round(1),
            "ad_spend": ad_spend.round(0).astype(int),
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "sales_data.csv": sales_data(),
        "retail_transactions.csv": retail_transactions(),
        "web_traffic.csv": web_traffic(),
    }
    for name, df in datasets.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df):,} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
