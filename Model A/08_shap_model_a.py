"""
08_shap_model_a.py

Deliverable I2 (Model A part): which features does the model rely on?

What this does:
  1. Repeats the same rolling walk-forward as 07 (same folds, same seed,
     same settings from config.py) for the Random Forest, the best
     Model A model
  2. In each fold, explains that fold's TEST weeks with TreeSHAP, using
     the model trained for that fold. So every SHAP value is
     out-of-sample, exactly like the predictions.
  3. Global importance: which features matter most overall
  4. Direction: for "buy" and "sell", does a high value of a feature
     push the prediction up or down?
  5. Importance over time, per feature and per data block (price, EIA,
     OPEC, rigs, macro, basis). Models B and C will produce the same
     block chart, so you can see whether geopolitics takes a share.

What SHAP is, in one line: for one prediction, it splits "how far this
prediction is from the average prediction" into a contribution from
each feature. Averaging the size of those contributions over all weeks
gives each feature's importance.

Caveat for the write-up: SHAP explains what the MODEL does, not what
drives oil prices. Model A only weakly beats chance, so these are the
patterns it leans on, not proven causes.

Install: pip install shap
"""

# %% 1. Imports and settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch

from config import CLASSES, MAIN_SEED, TEST_BLOCK, WINDOW_WEEKS, make_models

MODEL_NAME = "model_a"
EXPLAIN = "random_forest"   # best Model A model (see 07 results)
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results") / MODEL_NAME / "shap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Which file each feature came from, so features can be grouped into blocks
BLOCK_FILES = {
    "price": "brent_price_features_weekly.csv",
    "eia": "eia_supply_features_weekly.csv",
    "opec": "opec_features_weekly.csv",
    "rigs": "rig_features_weekly.csv",
    "macro": "macro_features_weekly.csv",
    "basis": "brent_basis_features_weekly.csv",
}


# %% 2. Load data
data = pd.read_csv(PROCESSED_DIR / f"{MODEL_NAME}_dataset.csv", parse_dates=["date"], index_col="date")
feature_cols = (PROCESSED_DIR / f"{MODEL_NAME}_feature_list.txt").read_text().split()
X = data[feature_cols]
y = data["target"].map({c: i for i, c in enumerate(CLASSES)})

block_of = {}
for block, file in BLOCK_FILES.items():
    for col in pd.read_csv(PROCESSED_DIR / file, nrows=0).columns:
        block_of[col] = block
block_of = {f: block_of.get(f, "other") for f in feature_cols}


# %% 3. Walk-forward with SHAP on each test block
# Identical folds to 07: rolling 260-week window, 26-week test blocks.
shap_parts = []   # one array per fold, shape (weeks, features, classes)
fold_of_week = []

for fold, start in enumerate(range(WINDOW_WEEKS, len(data), TEST_BLOCK), 1):
    end = min(start + TEST_BLOCK, len(data))
    X_train, y_train = X.iloc[start - WINDOW_WEEKS:start], y.iloc[start - WINDOW_WEEKS:start]
    X_test = X.iloc[start:end]

    model = make_models(MAIN_SEED)[EXPLAIN]
    model.fit(X_train, y_train)

    sv = shap.TreeExplainer(model).shap_values(X_test)
    # Older shap versions return a list (one array per class); newer ones
    # return one array (weeks, features, classes). Make it the latter.
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv
    shap_parts.append(sv)
    fold_of_week += [fold] * len(X_test)
    print(f"Fold {fold:2d}: explained {len(X_test)} test weeks")

shap_vals = np.concatenate(shap_parts)            # (weeks, features, classes)
test_index = X.index[WINDOW_WEEKS:]
X_test_all = X.loc[test_index]

# Save the raw values (long format) so Models B and C can be compared later
long = pd.concat({
    c: pd.DataFrame(shap_vals[:, :, i], index=test_index, columns=feature_cols)
    for i, c in enumerate(CLASSES)
}, names=["class", "date"])
long.to_csv(RESULTS_DIR / "shap_values.csv")


# %% 4. Global importance
# Mean |SHAP| over all test weeks, averaged over the three classes.
importance = pd.Series(np.abs(shap_vals).mean(axis=(0, 2)), index=feature_cols).sort_values(ascending=False)
importance_df = pd.DataFrame({"mean_abs_shap": importance, "block": importance.index.map(block_of)})
importance_df["share"] = importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum()
importance_df.to_csv(RESULTS_DIR / "global_importance.csv")
print("\nTop 15 features:")
print(importance_df.head(15).round(4))

block_share = importance_df.groupby("block")["share"].sum().sort_values(ascending=False)
print("\nShare of importance by data block:")
print(block_share.round(3))
block_share.to_csv(RESULTS_DIR / "block_share.csv")

colors = dict(zip(BLOCK_FILES, plt.cm.tab10.colors))
top = importance_df.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top.index, top["mean_abs_shap"], color=[colors.get(b, "grey") for b in top["block"]])
ax.legend(handles=[Patch(color=c, label=b) for b, c in colors.items()], title="data block", loc="lower right")
ax.set_xlabel("mean |SHAP| (average effect on predicted probability)")
ax.set_title(f"Model A ({EXPLAIN}): top 20 features, out-of-sample")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "global_importance.png", dpi=150)
plt.show()


# %% 5. Direction: beeswarm plots for "buy" and "sell"
# Each dot is one week. Position = how much that feature pushed the
# probability of this class up (right) or down (left). Colour = the
# feature's value that week (red high, blue low).
for cls in ["buy", "sell"]:
    i = CLASSES.index(cls)
    plt.figure()
    shap.summary_plot(shap_vals[:, :, i], X_test_all, max_display=15, show=False)
    plt.title(f'Model A: what pushes the "{cls}" probability')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"beeswarm_{cls}.png", dpi=150)
    plt.show()


# %% 6. Importance over time
# Mean |SHAP| per 6-month test block. If a feature only matters in
# certain periods (e.g. crises), it shows up here.
abs_week = pd.DataFrame(np.abs(shap_vals).mean(axis=2), index=test_index, columns=feature_cols)
abs_week["fold"] = fold_of_week
by_fold = abs_week.groupby("fold").mean()
by_fold.index = [test_index[np.array(fold_of_week) == f][0] for f in by_fold.index]
by_fold.to_csv(RESULTS_DIR / "importance_by_fold.csv")

# Block shares over time (each row sums to 1)
block_time = by_fold.T.groupby(by_fold.columns.map(block_of)).sum().T
block_time = block_time.div(block_time.sum(axis=1), axis=0)
block_time.to_csv(RESULTS_DIR / "block_share_by_fold.csv")

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for f in importance.index[:8]:
    axes[0].plot(by_fold.index, by_fold[f], marker=".", label=f)
axes[0].set_title("Top 8 features: importance per 6-month test block")
axes[0].legend(fontsize=8, ncol=2)
axes[1].stackplot(block_time.index, block_time.T.values, labels=block_time.columns,
                  colors=[colors.get(b, "grey") for b in block_time.columns])
axes[1].set_title("Share of importance by data block")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8, ncol=3, loc="upper left")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "importance_over_time.png", dpi=150)
plt.show()

print(f"\nEverything saved to {RESULTS_DIR}/")
