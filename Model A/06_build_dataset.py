"""
06_build_dataset.py

Final step for Model A's data: merge everything and build the target.

What this does:
  1. Reads the five weekly feature files from data/processed/
  2. Joins them on date, keeping only weeks where ALL sources have data
  3. Builds the target from your plan: Brent's return over the next
     five trading days -> buy (> +1.5%), sell (< -1.5%), or hold
  4. Checks for gaps, duplicates and lookahead
  5. Saves one table: data/processed/model_a_dataset.csv

Run 01-05 first. No downloads here, it only reads your saved files.
"""

#1. Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

THRESHOLD = 0.015   # +/-1.5%, from your project plan
HORIZON = 5         # trading days

FEATURE_FILES = {
    "price": "brent_price_features_weekly.csv",
    "eia": "eia_supply_features_weekly.csv",
    "opec": "opec_features_weekly.csv",
    "rigs": "rig_features_weekly.csv",
    "macro": "macro_features_weekly.csv",
}


# 2. Load and join the five feature files
parts = {}
for name, file in FEATURE_FILES.items():
    df = pd.read_csv(PROCESSED_DIR / file, parse_dates=["date"], index_col="date")
    if df.index.has_duplicates:
        raise SystemExit(f"{file} has duplicate dates - check that script")
    parts[name] = df
    print(f"{name:6s} {len(df):5d} weeks  {df.index.min().date()} to {df.index.max().date()}  "
          f"({df.shape[1]} columns)")

# The price file's "Close" is the raw $ price. It's needed for the target
# but is NOT a feature: a $ level means different things in 2008 and
# 2026, and the model would just memorise price eras. The ratio
# features (price vs moving averages etc.) already carry that info.
close_weekly = parts["price"].pop("Close")

# Inner join: keep only weeks where every source has a value.
features = pd.concat(parts.values(), axis=1, join="inner")

# If a script was run mid-week, its last row is an unfinished week.
features = features[features.index <= pd.Timestamp.today().normalize()]
print(f"\nAfter joining: {len(features)} weeks, {features.shape[1]} features")
print("Range:", features.index.min().date(), "to", features.index.max().date())


#3. Build the target from DAILY prices
# Your plan defines the target over the next FIVE TRADING DAYS, so it's
# built from the daily file, not by comparing one Friday to the next
# (holiday weeks would make those 4 days, not 5).
#
# For each week, find its last trading day (usually Friday, sometimes
# Thursday before a holiday), then look 5 trading days ahead.
daily = pd.read_csv(RAW_DIR / "brent_daily_raw.csv", parse_dates=["date"], index_col="date")
daily = daily[daily["Close"] > 0].sort_index()
closes = daily["Close"]

pos = closes.index.searchsorted(features.index, side="right") - 1  # last trading day <= Friday
valid = pos >= 0
origin_close = pd.Series(index=features.index, dtype=float)
future_close = pd.Series(index=features.index, dtype=float)
origin_close[valid] = closes.iloc[pos[valid]].values
has_future = valid & (pos + HORIZON < len(closes))
future_close[has_future] = closes.iloc[pos[has_future] + HORIZON].values

data = features.copy()
data["fwd_return_5d"] = future_close / origin_close - 1
data["target"] = pd.cut(
    data["fwd_return_5d"],
    bins=[-float("inf"), -THRESHOLD, THRESHOLD, float("inf")],
    labels=["sell", "hold", "buy"],
)

# Sanity check: the origin close should match the price file's Close.
mismatch = (origin_close - close_weekly.reindex(origin_close.index)).abs().max()
print(f"\nOrigin price vs price file, max difference: {mismatch:.6f} (should be 0)")


#4. Drop weeks without a target yet
# The most recent week has no "next 5 days" yet, so no label. Keep it
# aside: it's what you'd use to make a live prediction later.
latest = data[data["target"].isna()]
data = data.dropna(subset=["target"])
print(f"Weeks with a target: {len(data)}   (no target yet: {list(latest.index.date)})")


# 5. Checks
# Missing weeks: every Friday between start and end should be present.
all_fridays = pd.date_range(data.index.min(), data.index.max(), freq="W-FRI")
missing = all_fridays.difference(data.index)
print(f"\nMissing weeks: {len(missing)}", list(missing.date)[:10])

# Missing values inside the table
nan_cols = data.isna().sum()
print("Columns with missing values:", nan_cols[nan_cols > 0].to_dict() or "none")

# Class balance. If one class dominates, the model can look accurate
# just by always predicting it. Report this in your write-up.
counts = data["target"].value_counts().reindex(["sell", "hold", "buy"])
print("\nClass balance:")
print(pd.DataFrame({"weeks": counts, "share": (counts / len(data)).round(3)}))


#6. Save
out = PROCESSED_DIR / "model_a_dataset.csv"
data.to_csv(out)
latest.drop(columns=["fwd_return_5d", "target"]).to_csv(PROCESSED_DIR / "model_a_latest_week.csv")

feature_cols = [c for c in data.columns if c not in ("fwd_return_5d", "target")]
(PROCESSED_DIR / "model_a_feature_list.txt").write_text("\n".join(feature_cols))
print(f"\nSaved {out}: {len(data)} weeks x {len(feature_cols)} features + target")
print("Feature list saved to model_a_feature_list.txt")


# %% 7. Plots
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].hist(data["fwd_return_5d"], bins=80)
for x in (-THRESHOLD, THRESHOLD):
    axes[0].axvline(x, color="red", ls="--", lw=1)
axes[0].set_title("Next-5-day Brent returns (red lines = buy/sell thresholds)")

yearly = data.groupby(data.index.year)["target"].value_counts(normalize=True).unstack()[["sell", "hold", "buy"]]
yearly.plot(kind="bar", stacked=True, ax=axes[1], color=["#d62728", "#999999", "#2ca02c"])
axes[1].set_title("Class mix by year")
axes[1].legend(loc="upper left", ncol=3)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "model_a_target_plot.png", dpi=150)
plt.show()
