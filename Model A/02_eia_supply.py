"""
02_eia_supply.py

Block 2 of Model A: US supply data from the EIA.

What this does:
  1. Pulls two weekly series from the EIA API
       - US commercial crude inventories (excluding the SPR)
       - US crude oil field production
  2. Saves the untouched raw data to data/raw/
  3. Shifts each value to the week it was actually PUBLISHED (no lookahead)
  4. Builds features: levels, changes, and inventories vs their 5-year norm
  5. Saves the weekly feature table to data/processed/

Before running:
  1. Get a free API key: https://www.eia.gov/opendata/register.php
  2. In the terminal, before running the script:
       export EIA_API_KEY="your_key_here"
     (Keeps the key out of the code, so it never ends up on GitHub.)

Install:
  pip install requests pandas matplotlib
"""

# %% 1. Imports, folders and API key
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("EIA_API_KEY")
if not API_KEY:
    raise SystemExit('No API key found. Run: export EIA_API_KEY="your_key_here"')

# Start in 2000, not 2007: the "vs 5-year average" feature needs five
# years of history before your price data starts.
START = "2000-01-01"

# EIA series IDs (both weekly, from the Weekly Petroleum Status Report)
SERIES = {
    "inventories": "PET.WCESTUS1.W",  # commercial crude stocks ex-SPR, thousand barrels
    "production": "PET.WCRFPUS2.W",   # crude field production, thousand barrels/day
}


# %% 2. Pull the raw data and save it untouched
def fetch_eia(series_id: str) -> pd.Series:
    """Download one EIA series and return it as a date-indexed Series."""
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"api_key": API_KEY, "start": START, "length": 5000}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()  # stops with a clear error if the key/URL is wrong

    rows = resp.json()["response"]["data"]
    df = pd.DataFrame(rows)
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,
        index=pd.to_datetime(df["period"]),
    )
    return s.sort_index()


raw = pd.DataFrame({name: fetch_eia(sid) for name, sid in SERIES.items()})
raw.index.name = "week_ending"
raw.to_csv(RAW_DIR / "eia_supply_raw.csv")
print(f"Saved {len(raw)} rows to {RAW_DIR / 'eia_supply_raw.csv'}")


# %% 3. Inspect: what did we actually get?
print(raw.head())
print(raw.tail())
print("\nDate range:", raw.index.min().date(), "to", raw.index.max().date())
print("Missing values:\n", raw.isna().sum())

# Every row should be exactly 7 days apart. Anything else is a gap.
gaps = raw.index.to_series().diff().dt.days
print("\nGaps not equal to 7 days:\n", gaps[(gaps != 7) & gaps.notna()])

# EIA dates should all be Fridays (week ending). 4 = Friday.
print("Non-Friday dates:", (raw.index.dayofweek != 4).sum())

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
raw["inventories"].plot(ax=axes[0], title="US commercial crude inventories (k barrels)")
raw["production"].plot(ax=axes[1], title="US crude production (k barrels/day)")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "eia_raw_plot.png", dpi=150)
plt.show()


# %% 4. Shift to publication date (the important bit)
# The row labelled "week ending Friday 3rd" is NOT known on Friday 3rd.
# EIA publishes it the following Wednesday (the 8th). So on the Friday
# you make your forecast (the 10th), the latest number you'd actually
# have seen is for the week ending the 3rd.
#
# Shifting every date forward 7 days lines each value up with the first
# Friday it was publicly available. Holiday weeks sometimes push the
# release to Thursday, which is still before Friday, so this holds.
df = raw.copy()
df.index = df.index + pd.Timedelta(days=7)
df.index.name = "date"


# %% 5. Build features
inv = df["inventories"]
prod = df["production"]

# Inventories
df["inv_change_1w"] = inv.diff(1)            # weekly build (+) or draw (-)
df["inv_change_4w"] = inv.diff(4)            # last month's trend
# Inventories have a strong seasonal pattern (they build in spring,
# draw in summer driving season). Comparing to the same week in the
# last 5 years shows whether stocks are unusually tight or loose,
# which is what traders actually watch.
five_yr_avg = sum(inv.shift(52 * k) for k in range(1, 6)) / 5
df["inv_vs_5yr_avg"] = inv / five_yr_avg - 1

# Production
df["prod_change_4w"] = prod.pct_change(4)
df["prod_change_52w"] = prod.pct_change(52)  # year-on-year growth

df = df.rename(columns={"inventories": "inv_level", "production": "prod_level"})

feature_cols = [
    "inv_level", "inv_change_1w", "inv_change_4w", "inv_vs_5yr_avg",
    "prod_level", "prod_change_4w", "prod_change_52w",
]
weekly = df[feature_cols]


# %% 6. Trim to the price data's start and save
# The 5-year comparison leaves the first ~5 years empty, which is why
# we pulled from 2000. Now cut to match your price features.
weekly = weekly[weekly.index >= "2007-07-30"].dropna()
print(f"Weekly rows: {len(weekly)}")
print("Range:", weekly.index.min().date(), "to", weekly.index.max().date())

weekly.to_csv(PROCESSED_DIR / "eia_supply_features_weekly.csv")
print(f"Saved to {PROCESSED_DIR / 'eia_supply_features_weekly.csv'}")
print(weekly.describe().T[["mean", "std", "min", "max"]])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
weekly["inv_vs_5yr_avg"].plot(ax=axes[0], title="Inventories vs 5-year average")
axes[0].axhline(0, color="grey", lw=0.8)
weekly["prod_change_52w"].plot(ax=axes[1], title="Production, year-on-year change")
axes[1].axhline(0, color="grey", lw=0.8)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "eia_features_plot.png", dpi=150)
plt.show()




# to run this code you need this 
# export EIA_API_KEY="8FjQHCbNbqmfMgTwd68d8tD8m6KTp9KnwjmCLqc7"
# /usr/local/bin/python3 02_eia_supply.py