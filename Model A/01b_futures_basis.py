"""
01b_futures_basis.py

Extra block for Model A: the futures basis (backwardation / contango).

What it measures:
  basis = Brent spot price / Brent front-month futures price - 1

  Positive (spot above futures) = backwardation: buyers pay more for oil
  NOW than for later delivery, a sign the physical market is tight.
  Negative = contango: oil is plentiful, so storing it is paid for.

Where the data comes from:
  - Brent spot ("Europe Brent Spot Price FOB", Dated Brent) -> EIA API,
    series PET.RBRTE.D, the same API key as 02_eia_supply.py
  - Brent front-month futures -> data/raw/brent_daily_raw.csv (from 01)

Why not a proper curve spread (e.g. month 1 minus month 4)?
  EIA's free NYMEX futures series (contracts 1-4) stopped on 5 April
  2024, and free Brent curve data doesn't exist. Spot vs front month is
  the standard free proxy. If you ever pull CO1 and CO4 from Bloomberg,
  that would be the gold-standard replacement for this feature.

Before running:
  export EIA_API_KEY="your_key_here"   (same as for script 02)

Install: pip install requests pandas matplotlib
"""

# %% 1. Imports, folders and API key
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

API_KEY = os.environ.get("EIA_API_KEY")
if not API_KEY:
    raise SystemExit('No API key found. Run: export EIA_API_KEY="your_key_here"')

START = "2007-01-01"


# %% 2. Pull Brent spot from the EIA (daily)
# The API returns at most 5000 rows per request, and daily data since
# 2007 is about 5000 rows, so ask page by page until nothing comes back.
def fetch_eia_daily(series_id: str) -> pd.Series:
    rows, offset = [], 0
    while True:
        resp = requests.get(
            f"https://api.eia.gov/v2/seriesid/{series_id}",
            params={"api_key": API_KEY, "start": START, "length": 5000, "offset": offset},
            timeout=60,
        )
        resp.raise_for_status()
        page = resp.json()["response"]["data"]
        if not page:
            break
        rows += page
        offset += len(page)
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.to_datetime(df["period"]), name="spot")
    return s[~s.index.duplicated()].sort_index()


spot = fetch_eia_daily("PET.RBRTE.D")
spot.index.name = "date"
spot.to_csv(RAW_DIR / "brent_spot_raw.csv")
print(f"Brent spot: {len(spot)} days, {spot.index.min().date()} to {spot.index.max().date()}")


# %% 3. Line it up with the futures price from script 01
fut = pd.read_csv(RAW_DIR / "brent_daily_raw.csv", parse_dates=["date"], index_col="date")["Close"]

# Only days where BOTH prices exist (holidays differ slightly between
# the physical market and the exchange). Drop the April 2020 oddities
# and any zero or missing prices.
daily = pd.DataFrame({"spot": spot, "futures": fut}).dropna()
daily = daily[(daily["spot"] > 0) & (daily["futures"] > 0)]
daily["basis"] = daily["spot"] / daily["futures"] - 1

print(f"Days with both prices: {len(daily)}")
print(daily["basis"].describe())

# Timing note (for the write-up): Dated Brent is assessed and published
# every day at the market close, the same moment as the futures close.
# The EIA just republishes it later. So using a Friday's spot price in
# that Friday's forecast is information traders really had: no lookahead.


# %% 4. Weekly features
# Yahoo's BZ=F jumps to the next contract when the front month expires,
# which makes the daily basis jump for a day or two. Averaging over the
# week smooths those roll-day jumps out.
weekly = pd.DataFrame({
    # A week with no overlapping trading days (e.g. the 2009 Yahoo gap)
    # carries the previous week's value, up to 2 weeks, rather than
    # knocking that week out of the whole dataset.
    "basis_pct": daily["basis"].resample("W-FRI").mean().ffill(limit=2),
})
weekly["basis_change_4w"] = weekly["basis_pct"].diff(4)  # tightening or loosening?

weekly = weekly[weekly.index >= "2007-08-03"].dropna()
weekly = weekly[weekly.index <= pd.Timestamp.today().normalize()]
weekly.index.name = "date"
weekly.to_csv(PROCESSED_DIR / "brent_basis_features_weekly.csv")
print(f"\nSaved {len(weekly)} weeks, {weekly.index.min().date()} to {weekly.index.max().date()}")

# Weeks without a value (e.g. if the EIA is behind on recent weeks).
# The dataset script only keeps weeks every source has, so a lag here
# would cut the most recent weeks from the whole dataset.
last_week = pd.Timestamp.today().normalize() - pd.offsets.Week(weekday=4)
if weekly.index.max() < last_week - pd.Timedelta(days=14):
    print(f"WARNING: spot data ends {weekly.index.max().date()} - the EIA may be behind.")


# %% 5. Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(daily.index, daily["spot"], lw=0.8, label="spot (Dated Brent)")
axes[0].plot(daily.index, daily["futures"], lw=0.8, label="front-month futures")
axes[0].set_title("Brent spot vs futures ($/bbl)")
axes[0].legend()
axes[1].plot(weekly.index, weekly["basis_pct"] * 100)
axes[1].axhline(0, color="grey", lw=0.8)
axes[1].set_title("Basis, weekly average (%): above 0 = backwardation, below 0 = contango")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "brent_basis_plot.png", dpi=150)
plt.show()
