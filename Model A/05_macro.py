"""
05_macro.py

Block 4 of Model A: macro and market conditions.

Where each series comes from:
  - US Dollar Index (DXY)   -> Yahoo Finance (DX-Y.NYB)
  - S&P 500                 -> Yahoo Finance (^GSPC)
  - 10-year Treasury yield  -> FRED (DGS10)
  - VIX                     -> FRED (VIXCLS)
  - CPI inflation           -> FRED (CPIAUCNS)

Why not all from FRED? FRED doesn't carry the real DXY (only the Fed's
own dollar index), and its S&P 500 series only goes back 10 years
for licensing reasons. Yahoo has both in full.

FRED's CSV download needs no API key, so nothing to set up.

Install: pip install yfinance pandas requests matplotlib
"""

# 1. Imports and folders
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

START = "2005-01-01"  # a little before the price data, for warm-up


#2. Pull the daily series
def fetch_fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # first column is the date (FRED renamed it once, so don't rely on the name)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]), name=series_id)
    return s[s.index >= START]   # FRED marks missing days with "." -> NaN


def fetch_yahoo(ticker: str) -> pd.Series:
    df = yf.Ticker(ticker).history(start=START, interval="1d", auto_adjust=False)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df["Close"]


daily = pd.DataFrame({
    "dxy": fetch_yahoo("DX-Y.NYB"),
    "sp500": fetch_yahoo("^GSPC"),
    "us10y": fetch_fred("DGS10"),
    "vix": fetch_fred("VIXCLS"),
})
daily.index.name = "date"
daily.to_csv(RAW_DIR / "macro_daily_raw.csv")

cpi = fetch_fred("CPIAUCNS")  # monthly, dated the 1st of the month
cpi.index.name = "month"
cpi.to_csv(RAW_DIR / "cpi_monthly_raw.csv")
print(f"Saved {len(daily)} daily rows and {len(cpi)} CPI months to {RAW_DIR}")


#3. Inspect
print(daily.head())
print(daily.tail())
print("\nFirst valid date per series:\n", daily.apply(lambda c: c.first_valid_index()))
print("\nMissing values per series:\n", daily.isna().sum())
print("\nCPI range:", cpi.index.min().date(), "to", cpi.index.max().date())

fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
for ax, col in zip(axes[:4], daily.columns):
    ax.plot(daily.index, daily[col])
    ax.set_title(col)
axes[4].plot(cpi.index, cpi.values)
axes[4].set_title("CPI (index level)")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "macro_raw_plot.png", dpi=150)
plt.show()


#4. Daily features
# Each market trades on slightly different holidays, so carry the last
# known value forward for up to a few days rather than leaving gaps.
d = daily.ffill(limit=5)

f = pd.DataFrame(index=d.index)
# Dollar: oil is priced in dollars, so a stronger dollar tends to push
# oil down. Use % changes, not the level.
f["dxy_change_5d"] = d["dxy"].pct_change(5)
f["dxy_change_20d"] = d["dxy"].pct_change(20)
# Stocks: a read on risk appetite and the economy.
f["sp500_ret_5d"] = d["sp500"].pct_change(5)
f["sp500_ret_20d"] = d["sp500"].pct_change(20)
# 10-year yield: already a %, so use the level and its change in
# percentage points (diff, not pct_change).
f["us10y"] = d["us10y"]
f["us10y_change_20d"] = d["us10y"].diff(20)
# VIX: market fear gauge. Level matters (20 = normal, 40+ = panic).
f["vix"] = d["vix"]
f["vix_change_5d"] = d["vix"].diff(5)

# One row per week, Friday's value. All of these are known by the
# close on Friday, which is when the forecast is made.
weekly = f.resample("W-FRI").last()


#5. CPI: monthly, with publication lag
# CPI for a month is released around the 10th-15th of the NEXT month.
# To be safe, month M is only used from the 20th of month M+1.
#
# The NOT seasonally adjusted series (CPIAUCNS) is used on purpose:
# the seasonally adjusted one gets revised every year, so its history
# isn't what people actually saw at the time. Year-on-year inflation
# from the unadjusted index needs no seasonal adjustment anyway.
m = cpi.to_frame("cpi")
m["inflation_yoy"] = m["cpi"].pct_change(12)
m["inflation_change_3m"] = m["inflation_yoy"].diff(3)  # rising or falling?
m["available_from"] = m.index + pd.DateOffset(months=1, days=19)

weekly = pd.merge_asof(
    weekly.reset_index(),
    m[["inflation_yoy", "inflation_change_3m", "available_from"]].reset_index(drop=True),
    left_on="date", right_on="available_from", direction="backward",
).set_index("date").drop(columns="available_from")


# %% 6. Trim, save and check
weekly = weekly[weekly.index >= "2007-08-03"].dropna()
# If you run this mid-week, the current week isn't finished yet, so its
# "Friday" row would really hold e.g. Monday's values. Drop it.
weekly = weekly[weekly.index <= pd.Timestamp.today().normalize()]
weekly.to_csv(PROCESSED_DIR / "macro_features_weekly.csv")
print(f"\nWeekly rows: {len(weekly)}")
print("Range:", weekly.index.min().date(), "to", weekly.index.max().date())
print(weekly.describe().T[["mean", "std", "min", "max"]])

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(weekly.index, weekly["vix"]); axes[0].set_title("VIX")
axes[1].plot(weekly.index, weekly["us10y"]); axes[1].set_title("US 10-year yield (%)")
axes[2].plot(weekly.index, weekly["inflation_yoy"]); axes[2].set_title("CPI inflation, year-on-year (as known each Friday)")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "macro_features_plot.png", dpi=150)
plt.show()
