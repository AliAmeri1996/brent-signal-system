"""
03_opec_production.py

Block 3a of Model A: OPEC total crude production (monthly).

Your plan says "confirm OPEC production via EIA international data",
so this uses the same EIA API key as 02_eia_supply.py. No PDF scraping.

What this does:
  1. Pulls monthly OPEC crude production from the EIA API
  2. Saves the untouched raw data to data/raw/
  3. Moves each month to the date it was actually PUBLISHED (no lookahead)
  4. Spreads the monthly values onto your weekly Friday dates
  5. Builds features and saves them to data/processed/

Before running (same key as before):
  export EIA_API_KEY="your_key_here"
"""

#1. Imports, folders and settings
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

START = "2005-01-01"   # a bit before 2007 so the 12-month change has history
LAG_MONTHS = 3         # see cell 4

# Two possible EIA series for OPEC crude production. The script tries
# them in order and uses the first that works:
#   INTL = EIA international statistics (thousand barrels/day)
#   STEO = EIA Short-Term Energy Outlook (million barrels/day)
CANDIDATES = [
    ("INTL.57-1-OPEC-TBPD.M", 1.0),     # already thousand b/d
    ("STEO.COPR_OPEC.M", 1000.0),       # million b/d -> convert to thousand
]


#2. Pull the raw data and save it untouched
def fetch_eia(series_id: str) -> pd.Series:
    url = f"https://api.eia.gov/v2/seriesid/{series_id}"
    params = {"api_key": API_KEY, "start": START, "length": 5000}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()["response"]["data"]
    if not rows:
        raise ValueError("no rows returned")
    df = pd.DataFrame(rows)
    s = pd.Series(
        pd.to_numeric(df["value"], errors="coerce").values,
        index=pd.to_datetime(df["period"]),  # "2024-03" -> 2024-03-01
    )
    return s.sort_index()


raw = None
for series_id, scale in CANDIDATES:
    try:
        raw = fetch_eia(series_id) * scale
        print(f"Using series {series_id}")
        break
    except Exception as e:
        print(f"{series_id} failed ({e}), trying next...")

if raw is None:
    raise SystemExit("Neither OPEC series worked. Send Claude the error messages above.")

# STEO also contains FORECASTS for future months. Drop anything that
# hasn't happened yet.
raw = raw[raw.index <= pd.Timestamp.today()]
raw.name = "opec_prod"
raw.index.name = "month"
raw.to_csv(RAW_DIR / "opec_production_raw.csv")
print(f"Saved {len(raw)} months to {RAW_DIR / 'opec_production_raw.csv'}")


# %% 3. Inspect
print(raw.head())
print(raw.tail())
print("\nRange:", raw.index.min().date(), "to", raw.index.max().date())
print("Missing values:", raw.isna().sum())
gaps = raw.index.to_series().diff().dt.days
print("Gaps over 31 days:\n", gaps[gaps > 31])

raw.plot(title="OPEC crude production (thousand barrels/day)", figsize=(10, 4))
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "opec_raw_plot.png", dpi=150)
plt.show()


# %% 4. Shift to publication date
# The figure for January is not known on 1 February. Monthly production
# estimates come out with a delay, so the January number is treated as
# available from 1 April (LAG_MONTHS = 3).
#
# 3 months is deliberately cautious. Being too cautious only makes the
# feature a bit staler. Being too optimistic leaks future information,
# which is the mistake examiners look for. Check EIA's release schedule
# for the series you end up using and cite it in your methodology.
monthly = raw.to_frame()
monthly["available_from"] = monthly.index + pd.DateOffset(months=LAG_MONTHS)

# Features calculated on the monthly series itself, before going weekly
monthly["opec_prod_change_1m"] = monthly["opec_prod"].pct_change(1)
monthly["opec_prod_change_12m"] = monthly["opec_prod"].pct_change(12)


# %% 5. Spread onto weekly Fridays
# For each Friday, take the most recent month that was already published.
# merge_asof does exactly this: "latest row on or before this date".
fridays = pd.DataFrame(
    {"date": pd.date_range("2007-08-03", pd.Timestamp.today(), freq="W-FRI")}
)
weekly = pd.merge_asof(
    fridays,
    monthly.reset_index().sort_values("available_from"),
    left_on="date",
    right_on="available_from",
    direction="backward",
)

# How stale is the number? Useful for the model and for your write-up.
weekly["opec_weeks_since_update"] = (
    (weekly["date"] - weekly["available_from"]).dt.days // 7
)

weekly = weekly.set_index("date")[
    ["opec_prod", "opec_prod_change_1m", "opec_prod_change_12m", "opec_weeks_since_update"]
].dropna()


# %% 6. Save and check
weekly.to_csv(PROCESSED_DIR / "opec_features_weekly.csv")
print(f"Weekly rows: {len(weekly)}")
print("Range:", weekly.index.min().date(), "to", weekly.index.max().date())
print(weekly.describe().T[["mean", "std", "min", "max"]])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
weekly["opec_prod"].plot(ax=axes[0], title="OPEC production as known each Friday")
weekly["opec_prod_change_12m"].plot(ax=axes[1], title="OPEC production, 12-month change")
axes[1].axhline(0, color="grey", lw=0.8)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "opec_features_plot.png", dpi=150)
plt.show()



#to run this 
# export EIA_API_KEY="8FjQHCbNbqmfMgTwd68d8tD8m6KTp9KnwjmCLqc7"
# /usr/local/bin/python3 03_opec_production.py