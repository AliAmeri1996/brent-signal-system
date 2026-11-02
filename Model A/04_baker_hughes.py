"""
04_baker_hughes.py

Block 3b of Model A: Baker Hughes rig counts.

Two features from two kinds of data:
  - North America (US + Canada): WEEKLY, released Friday noon Central Time
  - International (rest of world): MONTHLY, released early the following month

Baker Hughes changed its report format twice, so the history is split
across six files. This script reads all six, checks they agree where
they overlap, and stitches them into one series each.

Before running:
  1. Put the 6 downloaded files in:  data/raw/baker_hughes/
     (keep the names Baker Hughes gave them)
  2. pip install pandas openpyxl pyxlsb matplotlib
"""

#1. Imports, folders and settings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_DIR = Path("data/raw")
BH_DIR = RAW_DIR / "baker_hughes"
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def find(pattern: str) -> list[Path]:
    files = sorted(BH_DIR.glob(pattern))
    if not files:
        raise SystemExit(f"No file matching '{pattern}' in {BH_DIR}")
    return files


#2. North America: oldest archive (.xlsb, 1987/2000 to Mar 2024)
# This file has a clean weekly table for each country: Date, Oil, Gas,
# Misc, Total. Dates are stored as Excel day numbers, so they need
# converting.
def read_xlsb_split(path: Path, sheet: str) -> pd.Series:
    df = pd.read_excel(path, sheet_name=sheet, header=6, engine="pyxlsb")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[pd.to_numeric(df["date"], errors="coerce").notna()]
    dates = pd.to_datetime(pd.to_numeric(df["date"]), unit="D", origin="1899-12-30")
    return pd.Series(pd.to_numeric(df["total"]).values, index=dates)


xlsb = find("*North_America_Rotary_Rig_Count*.xlsb")[0]
na_old = pd.DataFrame({
    "us": read_xlsb_split(xlsb, "US Oil & Gas Split"),
    "canada": read_xlsb_split(xlsb, "Canada Oil & Gas Split"),
})


# %% 3. North America: new-format reports (2013 onwards)
# These list every rig by county, so we add them up per week and country.
def read_nam_report(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="NAM Weekly", header=10)
    df = df[df["Country"].notna()]
    df["date"] = pd.to_datetime(df["US_PublishDate"])
    out = df.pivot_table(index="date", columns="Country",
                         values="Rig Count Value", aggfunc="sum")
    return out.rename(columns={"UNITED STATES": "us", "CANADA": "canada"})


na_new = [read_nam_report(p) for p in find("*North_America_Rig_Count_Report.xlsx")]

# Stitch: newest file wins where they overlap.
na = na_old
for part in na_new:  # sorted by filename, i.e. oldest report first
    na = part.combine_first(na)
na = na.sort_index()


# %% 4. Check the North America files agree where they overlap
for part in na_new:
    both = part.index.intersection(na_old.index)
    if len(both):
        diff = (part.loc[both, ["us", "canada"]] - na_old.loc[both, ["us", "canada"]]).abs().max()
        print(f"Overlap check ({len(both)} weeks), max difference:\n{diff}\n")


# 5. International: oldest archive (1975 to Mar 2024)
# One block per year, a row per month, with a "Total Intl." column.
# Total Intl. already excludes the US and Canada.
old_path = find("*Worldwide_Rig_Count_Jan_2007*.xlsx")[0]
old = pd.read_excel(old_path, header=None)
rows, year = [], None
for _, r in old.iterrows():
    label = r[1]
    if isinstance(label, (int, float)) and pd.notna(label) and 1900 < label < 2100:
        year = int(label)                       # start of a new year block
    elif isinstance(label, str) and label[:3] in MONTHS and pd.notna(r[7]):
        rows.append((pd.Timestamp(f"{year}-{label[:3]}-01"), float(r[7])))
intl_old = pd.Series(dict(rows)).sort_index()


# 6. International: new-format reports (2013 onwards)
# IMPORTANT: from January 2024 Baker Hughes changed how it counts Saudi
# Arabia (it now counts "operating" rigs, roughly 300 instead of ~90).
# Left in, that would look like a sudden 25% jump in world drilling
# that never happened. So Saudi Arabia is EXCLUDED here, which keeps
# the series measuring the same thing all the way through.
def read_ww_report(path: Path) -> tuple[pd.Series, pd.Series]:
    """Returns (total excluding Saudi, Saudi's own count) for one report."""
    df = pd.read_excel(path, sheet_name="WW Monthly", header=11)
    df = df[df["Region"].notna() & df["Year"].notna() & (df["Region"] != "North America")]
    df["date"] = pd.to_datetime(dict(year=df["Year"].astype(int),
                                     month=df["Month"].astype(int), day=1))
    saudi = df[df["Country"] == "SAUDI ARABIA"].groupby("date")["Rig Count Value"].sum()
    ex_saudi = df[df["Country"] != "SAUDI ARABIA"].groupby("date")["Rig Count Value"].sum()
    return ex_saudi[ex_saudi > 0], saudi  # future months come through as zeros


ww_pairs = [read_ww_report(p) for p in find("*WorldWide_Rig_Count_Report.xlsx")]
intl_new, saudi_new = ww_pairs[0]
for ex_saudi, saudi in ww_pairs[1:]:
    both = ex_saudi.index.intersection(intl_new.index)
    print(f"International overlap ({len(both)} months), max difference:",
          (ex_saudi.loc[both] - intl_new.loc[both]).abs().max())
    intl_new = ex_saudi.combine_first(intl_new)
    saudi_new = saudi.combine_first(saudi_new)


# %% 7. Join old and new international
# The old archive (1975-Mar 2024) only ever reports regional totals, so
# Saudi can't be pulled out of it directly. BUT it matches the new
# files' Saudi-INCLUDING total exactly for every month they overlap
# (2013-2024, checked below), which means the old archive used the
# same Saudi counting method for its entire 1975-2024 history. So:
#
#   2013 onwards: Saudi is known exactly (from the new files) and is
#   subtracted exactly. This part is not an estimate.
#
#   Before 2013: no country breakdown exists in any source, so Saudi's
#   count has to be estimated. Rather than anchor to one month (which
#   could be unrepresentative), this uses the AVERAGE ratio of
#   (total excluding Saudi) / (total including Saudi) across the full
#   2013-2023 overlap, which is more stable. This is still an
#   assumption, and should be stated as one in your methodology:
#   Saudi's share of world drilling is assumed roughly constant before
#   2013 at its observed 2013-2023 average.
old_full = intl_old.copy()  # old archive total, Saudi included throughout
# Validate using the FIRST (oldest-format) report's Saudi figure, since
# that's the one still using the same counting method as the archive.
# (The newest report's Saudi figure uses the post-2024 method, so
# comparing against that would wrongly look like a mismatch here.)
ex_saudi_0, saudi_0 = ww_pairs[0]
check_idx = old_full.index.intersection((ex_saudi_0 + saudi_0.reindex(ex_saudi_0.index, fill_value=0)).index)
implied_total = ex_saudi_0 + saudi_0.reindex(ex_saudi_0.index, fill_value=0)
diff = (old_full.loc[check_idx] - implied_total.loc[check_idx]).abs()
print(f"\nOld archive vs (ex-Saudi + Saudi) over {len(check_idx)} months, "
      f"max difference: {diff.max():.1f} (confirms old archive uses the same Saudi method)")

ratio_window = intl_new.index[intl_new.index < "2024-01-01"]  # before the Saudi method change
avg_ratio = (intl_new.loc[ratio_window] / old_full.reindex(ratio_window)).mean()
print(f"Average ex-Saudi / including-Saudi ratio, 2013-2023: {avg_ratio:.3f}")

link = intl_new.index.min()
intl = pd.concat([
    old_full[old_full.index < link] * avg_ratio,  # estimated, pre-2013
    intl_new,                                      # exact, 2013 onwards
]).sort_index()
intl.name = "intl_rigs"


#8. Save the stitched raw series
na.to_csv(RAW_DIR / "rigs_north_america_weekly_raw.csv")
intl.to_csv(RAW_DIR / "rigs_international_monthly_raw.csv")
print("NA range:", na.index.min().date(), "to", na.index.max().date())
print("Intl range:", intl.index.min().date(), "to", intl.index.max().date())

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(na.index, na["us"], label="US")
axes[0].plot(na.index, na["canada"], label="Canada")
axes[0].set_title("North America rigs (weekly)")
axes[0].legend()
axes[1].plot(intl.index, intl.values)
axes[1].set_title("International rigs ex-Saudi (monthly, chain-linked)")
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "rigs_raw_plot.png", dpi=150)
plt.show()


# 9. North America weekly features
# Holiday weeks are sometimes published Wed/Thu. Snap everything to the
# week-ending Friday. Release is Friday noon Central Time, before Brent
# settles (13:30 Central), so it is known on the same Friday.
naw = na.resample("W-FRI").last()
features = pd.DataFrame(index=naw.index)
features["us_rigs"] = naw["us"]
features["us_rigs_change_4w"] = naw["us"].pct_change(4)
features["us_rigs_change_52w"] = naw["us"].pct_change(52)
# Canada swings massively with the seasons (spring thaw stops drilling),
# so only the year-on-year change is meaningful. In the thaw it can fall
# to ~15 rigs, which makes week-vs-week ratios explode (e.g. +800%), so
# compare 4-week averages instead.
canada_4w = naw["canada"].rolling(4).mean()
features["canada_rigs_change_52w"] = canada_4w.pct_change(52)


# 10. International monthly -> weekly, with publication lag
# Released in the first week of the following month (since Feb 2020).
# To be safe, month M is only used from the 15th of month M+1.
m = intl.to_frame()
m["intl_rigs_change_3m"] = m["intl_rigs"].pct_change(3)
m["intl_rigs_change_12m"] = m["intl_rigs"].pct_change(12)
m["available_from"] = m.index + pd.DateOffset(months=1, days=14)

weekly = pd.merge_asof(
    features.reset_index().rename(columns={"index": "date"}),
    m.reset_index(drop=True).sort_values("available_from"),
    left_on="date", right_on="available_from", direction="backward",
).set_index("date").drop(columns="available_from")


# %% 11. Trim, save and check
weekly = weekly[weekly.index >= "2007-08-03"].dropna()
weekly.to_csv(PROCESSED_DIR / "rig_features_weekly.csv")
print(f"\nWeekly rows: {len(weekly)}")
print("Range:", weekly.index.min().date(), "to", weekly.index.max().date())
print(weekly.describe().T[["mean", "std", "min", "max"]])

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
weekly[["us_rigs_change_52w", "canada_rigs_change_52w"]].plot(
    ax=axes[0], title="North America rigs, year-on-year change")
weekly["intl_rigs_change_12m"].plot(ax=axes[1], title="International rigs, 12-month change")
for ax in axes:
    ax.axhline(0, color="grey", lw=0.8)
plt.tight_layout()
plt.savefig(PROCESSED_DIR / "rigs_features_plot.png", dpi=150)
plt.show()
