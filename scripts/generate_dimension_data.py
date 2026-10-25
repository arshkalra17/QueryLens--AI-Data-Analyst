"""Generate the dimension tables that sales_data.csv joins to.

- sample_data/sales_customers.csv : one row per customer_id in sales_data.csv
- sample_data/product_catalog.csv : one row per product_category

Deterministic (fixed seed) so benchmark gold answers never change:

    python scripts/generate_dimension_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data"


def main() -> None:
    rng = np.random.default_rng(42)
    ids = [f"CUS{i:04d}" for i in range(1, 121)]
    customers = pd.DataFrame({
        "customer_id": ids,
        "customer_name": [f"Account {i:03d}" for i in range(1, 121)],
        "segment": rng.choice(["Enterprise", "SMB", "Consumer"], len(ids), p=[0.25, 0.45, 0.30]),
        "country": rng.choice(["India", "USA", "UK", "Germany", "Japan"], len(ids), p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        "signup_date": pd.to_datetime("2021-01-01")
        + pd.to_timedelta(rng.integers(0, 3 * 365, len(ids)), unit="D"),
        "credit_limit": rng.integers(5, 101, len(ids)) * 1000,
    })
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")
    customers.to_csv(OUT / "sales_customers.csv", index=False)

    catalog = pd.DataFrame({
        "product_category": ["Charger", "Headset", "Keyboard", "Laptop", "Monitor", "Mouse", "Phone", "Tablet"],
        "product_line": ["Mobile", "Audio", "Peripherals", "Computing", "Computing", "Peripherals", "Mobile", "Mobile"],
        "brand": ["Voltix", "SoundCore", "TypeRight", "NovaBook", "ClearView", "TypeRight", "NovaPhone", "NovaBook"],
        "warranty_years": [1, 1, 2, 3, 3, 2, 2, 2],
        "launch_year": [2019, 2020, 2018, 2021, 2020, 2018, 2022, 2021],
    })
    catalog.to_csv(OUT / "product_catalog.csv", index=False)
    print(f"wrote {OUT / 'sales_customers.csv'} ({len(customers)} rows) and {OUT / 'product_catalog.csv'} ({len(catalog)} rows)")


if __name__ == "__main__":
    main()
