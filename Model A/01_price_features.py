"""
01_price_features.py

Block 1 of Model A: Brent price and technical features.

What this does:
  1. Pulls daily Brent futures (BZ=F) from Yahoo Finance
  2. Saves the untouched raw data to data/raw/
  3. Checks the data for gaps and bad values
  4. Computes returns and technical indicators on DAILY data
  5. Resamples to WEEKLY (last trading day of each week)
  6. Saves the weekly feature table to data/processed/


Install first (inside your virtual environment):
  pip install yfinance pandas ta matplotlib
"""

#1. Imports and folders
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TICKER = "BZ=F"  # Brent crude front-month continuous futures on Yahoo


# 2. Pull raw daily data and save it untouched
# Ticker().history() returns flat columns (yf.download can return a
# multi-level header, which is annoying to work with).
raw = yf.Ticker(TICKER).history(period="max", interval="1d", auto_adjust=False)

# Yahoo dates come with a New York timezone attached. Strip it so the
# dates line up cleanly with EIA/FRED data later.
raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
raw.index.name = "date"
raw = raw[["Open", "High", "Low", "Close", "Volume"]]

raw.to_csv(RAW_DIR / "brent_daily_raw.csv")
print(f"Saved {len(raw)} rows to {RAW_DIR / 'brent_daily_raw.csv'}")


# 3. Inspect: what did we actually get?
print(raw.head())
print(raw.tail())
print("\nDate range:", raw.index.min().date(), "to", raw.index.max().date())
print("Rows:", len(raw))
print("\nMissing values per column:\n", raw.isna().sum())
print("\nZero or negative closes:", (raw["Close"] <= 0).sum())

# Normal gaps are weekends/holidays (up to ~4 days). Anything longer
# is worth looking at.
gaps = raw.index.to_series().diff().dt.days
print("\nGaps longer than 4 days:\n", gaps[gaps > 4])

raw["Close"].plot(title="Brent front-month close (raw)", figsize=(10, 4))
plt.savefig(PROCESSED_DIR / "Brent front-month close .png", dpi=150)
plt.show()


# 4. Basic cleaning
df = raw.dropna(subset=["Close"]).copy()
df = df[df["Close"] > 0]                              # drop bad prints
df = df[~df.index.duplicated(keep="last")].sort_index()
close = df["Close"]


# 5. Returns (1, 5 and 20 trading days)
# pct_change(n) = close_today / close_n_days_ago - 1
# Only uses past prices, so no lookahead.
df["ret_1d"] = close.pct_change(1)
df["ret_5d"] = close.pct_change(5)
df["ret_20d"] = close.pct_change(20)


# 6. Technical indicators
# These all need DAILY data to mean anything (e.g. RSI-14 = 14 trading
# days), which is why we compute them before going weekly.

# RSI: momentum oscillator, 0-100. Above 70 = "overbought", below 30 = "oversold".
df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

# MACD: gap between a fast (12) and slow (26) moving average, plus a
# 9-day signal line. Divided by price so a $2 gap means the same thing
# at $20 oil as at $100 oil.
macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
df["macd_pct"] = macd.macd() / close
df["macd_signal_pct"] = macd.macd_signal() / close
df["macd_hist_pct"] = macd.macd_diff() / close

# Bollinger Bands: 20-day average +/- 2 standard deviations.
# B = where price sits inside the bands (0 = lower band, 1 = upper band).
# Width = how wide the bands are, i.e. a volatility measure.
bb = BollingerBands(close=close, window=20, window_dev=2)
df["bb_pctb"] = bb.bollinger_pband()
df["bb_width"] = bb.bollinger_wband()

# ATR: average daily trading range over 14 days (volatility).
# Also expressed as % of price so it is comparable across decades.
atr = AverageTrueRange(high=df["High"], low=df["Low"], close=close, window=14)
df["atr_14_pct"] = atr.average_true_range() / close

# Moving averages: rather than the raw $ level, use how far price is
# above/below each average. Tree models split on values, and a raw
# "MA = $85" means nothing on its own across 20 years of prices.
for w in (20, 50, 200):
    ma = close.rolling(w).mean()
    df[f"price_vs_ma_{w}"] = close / ma - 1

# Trend structure: is the short average above the long one?
df["ma_50_vs_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1


#7. Resample to weekly
# One row per week, labelled by the Friday, holding the value from the
# last trading day of that week. That is exactly what you would have
# known at the end of the week, so no lookahead.
feature_cols = [
    "ret_1d", "ret_5d", "ret_20d",
    "rsi_14",
    "macd_pct", "macd_signal_pct", "macd_hist_pct",
    "bb_pctb", "bb_width",
    "atr_14_pct",
    "price_vs_ma_20", "price_vs_ma_50", "price_vs_ma_200",
    "ma_50_vs_200",
]

weekly = df[["Close"] + feature_cols].resample("W-FRI").last()

# The 200-day average needs ~200 trading days of warm-up, so the first
# ~10 months have empty values. Drop them.
before = len(weekly)
weekly = weekly.dropna()
print(f"Weekly rows: {len(weekly)} (dropped {before - len(weekly)} warm-up/empty weeks)")
print("Weekly range:", weekly.index.min().date(), "to", weekly.index.max().date())


#8. Save and sanity check
weekly.to_csv(PROCESSED_DIR / "brent_price_features_weekly.csv")
print(f"Saved to {PROCESSED_DIR / 'brent_price_features_weekly.csv'}")

print(weekly.describe().T[["mean", "std", "min", "max"]])

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
weekly["Close"].plot(ax=axes[0], title="Weekly close")
weekly["rsi_14"].plot(ax=axes[1], title="RSI 14")
weekly["atr_14_pct"].plot(ax=axes[2], title="ATR as % of price")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "close_rsi_atr_plot.png", dpi=150)
plt.show()


#9. Not done here: futures basis (backwardation/contango)
# Basis needs TWO contracts at once (front month vs a later month).
# BZ=F is only the front month, and Yahoo drops expired contracts, so
# it can't give a clean historical second contract. This gets its own
# script once a source for the second contract is sorted.
