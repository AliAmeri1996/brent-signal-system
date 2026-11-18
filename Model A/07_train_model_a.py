"""
07_train_model_a.py

Deliverable B2: train and evaluate Model A with walk-forward backtesting.

What this does:
  1. Loads model_a_dataset.csv and reports the class split
  2. Walk-forward with a ROLLING window: train on the most recent 260
     weeks, predict the next 26, slide forward 26 weeks, repeat.
     The model never sees the future.
  3. Trains Random Forest, XGBoost and LightGBM (balanced classes) on
     identical folds, plus an ensemble that averages their probabilities
  4. Baselines on the same weeks:
       - always "hold"
       - majority class of the training window
       - last week's class (persistence)
       - climatology: the training window's class frequencies as
         probabilities (the probability version of "majority")
  5. Scores: accuracy, macro-F1, Brier score, confusion matrices
  6. Tests:
       - Diebold-Mariano (HLN-corrected) on hit/miss loss vs the best baseline
       - Diebold-Mariano on Brier loss vs climatology
       - Pesaran-Timmermann (2009): is predicted direction related to
         actual direction at all?
  7. Repeats the whole thing with 5 seeds to show how much is noise
  8. Saves everything to results/model_a/

All frozen settings live in config.py, shared with Models B and C.

Install:
  pip install scikit-learn xgboost lightgbm scipy matplotlib pandas
"""

# %% 1. Imports and settings
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from config import CLASSES, MAIN_SEED, SEEDS, TEST_BLOCK, WINDOW_WEEKS, make_models

MODEL_NAME = "model_a"
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results") / MODEL_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
MODELS = ["random_forest", "xgboost", "lightgbm", "ensemble"]
BASELINES = ["baseline_always_hold", "baseline_majority", "baseline_last_week", "baseline_climatology"]


# %% 2. Load data and report the class split
data = pd.read_csv(PROCESSED_DIR / f"{MODEL_NAME}_dataset.csv", parse_dates=["date"], index_col="date")
feature_cols = (PROCESSED_DIR / f"{MODEL_NAME}_feature_list.txt").read_text().split()

X = data[feature_cols]
y = data["target"].map(CLASS_TO_INT)
print(f"{len(data)} weeks, {len(feature_cols)} features")

# Class split at +/-1.5%. This is the evidence for the threshold: it
# splits the weeks into roughly equal thirds, so no class dominates and
# a model can't look good by always predicting one answer.
test_start = data.index[WINDOW_WEEKS]
class_split = pd.DataFrame({
    "all_weeks": data["target"].value_counts(normalize=True),
    "test_weeks": data.loc[test_start:, "target"].value_counts(normalize=True),
}).reindex(CLASSES).round(3)
print("\nClass split:\n", class_split)
class_split.to_csv(RESULTS_DIR / "class_split.csv")


# %% 3. Walk-forward loop (rolling window)
# Fold 1: train weeks   0-259  -> test weeks 260-285
# Fold 2: train weeks  26-285  -> test weeks 286-311
# Fold 3: train weeks  52-311  -> test weeks 312-337
# ...the training window always has exactly 260 weeks.
#
# No gap is needed between train and test: the last training label
# (week t's next-5-day return) is fully known by the close of week t+1,
# which is exactly when the first test prediction is made.
def walk_forward(seed: int, verbose: bool = False) -> pd.DataFrame:
    rows = []
    for fold, start in enumerate(range(WINDOW_WEEKS, len(data), TEST_BLOCK), 1):
        end = min(start + TEST_BLOCK, len(data))
        X_train, y_train = X.iloc[start - WINDOW_WEEKS:start], y.iloc[start - WINDOW_WEEKS:start]
        X_test, y_test = X.iloc[start:end], y.iloc[start:end]

        probs = {}
        for name, model in make_models(seed).items():
            if name == "xgboost":
                # XGBoost has no class_weight option, so pass balanced weights here
                model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
            else:
                model.fit(X_train, y_train)
            probs[name] = model.predict_proba(X_test)
        probs["ensemble"] = np.mean([probs["random_forest"], probs["xgboost"], probs["lightgbm"]], axis=0)

        # Climatology: class frequencies in the training window, same every week of the fold
        freq = y_train.value_counts(normalize=True).reindex(range(len(CLASSES)), fill_value=0).values
        probs["baseline_climatology"] = np.tile(freq, (len(X_test), 1))

        fold_df = pd.DataFrame(index=X_test.index)
        fold_df["fold"] = fold
        fold_df["actual"] = y_test.values
        for name, p in probs.items():
            fold_df[name] = p.argmax(axis=1)
            for i, c in enumerate(CLASSES):
                fold_df[f"{name}_p_{c}"] = p[:, i]
        fold_df["baseline_always_hold"] = CLASS_TO_INT["hold"]
        fold_df["baseline_majority"] = y_train.mode()[0]
        rows.append(fold_df)

        if verbose:
            print(f"Fold {fold:2d}: train {X_train.index[0].date()} - {X_train.index[-1].date()}, "
                  f"test {X_test.index[0].date()} - {X_test.index[-1].date()} ({len(X_test)} weeks)")

    preds = pd.concat(rows)
    # Persistence: last week's label is known by this week's close, so it's a fair real-time baseline
    preds["baseline_last_week"] = y.shift(1).reindex(preds.index).astype(int)
    return preds


def brier(preds: pd.DataFrame, name: str) -> np.ndarray:
    """Multi-class Brier score per week: squared distance between the
    predicted probabilities and the actual outcome (0 = perfect, 2 = worst).
    Unlike hit/miss, it rewards a model for being confident when right
    and penalises it for being confident when wrong."""
    p = preds[[f"{name}_p_{c}" for c in CLASSES]].values
    onehot = np.eye(len(CLASSES))[preds["actual"].values]
    return ((p - onehot) ** 2).sum(axis=1)


def score_table(preds: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for name in MODELS + BASELINES:
        has_probs = f"{name}_p_sell" in preds
        out[name] = {
            "accuracy": accuracy_score(preds["actual"], preds[name]),
            # Macro-F1 averages the F1 of each class equally, so a model
            # can't score well by ignoring one class. Headline metric.
            "macro_f1": f1_score(preds["actual"], preds[name], average="macro"),
            # Only for forecasts that give probabilities
            "brier": brier(preds, name).mean() if has_probs else np.nan,
        }
    return pd.DataFrame(out).T.sort_values("macro_f1", ascending=False)


preds = walk_forward(MAIN_SEED, verbose=True)
preds.to_csv(RESULTS_DIR / "predictions.csv")
print(f"\nOut-of-sample weeks: {len(preds)} ({preds.index[0].date()} to {preds.index[-1].date()})")


# %% 4. Scores
scores = score_table(preds)
print(f"\nOut-of-sample scores (seed {MAIN_SEED}):")
print(scores.round(3))
scores.to_csv(RESULTS_DIR / "scores.csv")


# %% 5. Diebold-Mariano tests
# Is a model's loss significantly lower than a baseline's, or could the
# difference be luck? d = baseline loss - model loss; if the model is
# better, d is positive on average. One-sided, with the Harvey,
# Leybourne & Newbold (1997) small-sample correction.
def diebold_mariano(loss_base: np.ndarray, loss_model: np.ndarray, h: int = 1) -> tuple[float, float]:
    d = loss_base - loss_model
    n = len(d)
    d_mean = d.mean()
    gamma = [np.mean((d[k:] - d_mean) * (d[:n - k] - d_mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d_mean / np.sqrt(var)
    dm_hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm_hln, 1 - stats.t.cdf(dm_hln, df=n - 1)


# Two versions:
#  - hit/miss loss vs the best baseline. Simple, but it throws away the
#    probabilities, so it has little power to detect small differences.
#  - Brier loss vs climatology. Uses the full probabilities, so it's the
#    more sensitive test. Climatology is the fair reference because it's
#    the best you can do with probabilities and no features at all.
best_baseline = scores.loc[["baseline_always_hold", "baseline_majority", "baseline_last_week"], "accuracy"].idxmax()
hit_loss_base = (preds[best_baseline] != preds["actual"]).astype(float).values
brier_base = brier(preds, "baseline_climatology")

dm_rows = []
print(f"\nDiebold-Mariano (one-sided, HLN-corrected):")
for name in MODELS:
    hit_loss = (preds[name] != preds["actual"]).astype(float).values
    for loss_type, base_name, lb, lm in [
        ("hit_miss", best_baseline, hit_loss_base, hit_loss),
        ("brier", "baseline_climatology", brier_base, brier(preds, name)),
    ]:
        stat, p = diebold_mariano(lb, lm)
        dm_rows.append({"model": name, "loss": loss_type, "vs": base_name, "dm_stat": stat, "p_value": p})
        # p near 1 means the BASELINE is significantly better than the model
        verdict = "model better (5%)" if p < 0.05 else "model WORSE (5%)" if p > 0.95 else "no significant difference"
        print(f"  {name:14s} {loss_type:8s} vs {base_name:22s} DM = {stat:6.2f}  p = {p:.3f}  ({verdict})")
pd.DataFrame(dm_rows).to_csv(RESULTS_DIR / "diebold_mariano.csv", index=False)


# %% 6. Pesaran-Timmermann test (2009)
# A different question from DM: is the predicted class related to the
# actual class AT ALL? Under "no relation", knowing the prediction tells
# you nothing about the outcome.
#
# How: regress dummies for the actual class on dummies for the predicted
# class and test that all slopes are zero. Standard errors are
# Newey-West (HAC), which allows for weeks being correlated, as in
# Pesaran & Timmermann (2009). The Pearson chi-squared test on the
# confusion matrix is reported alongside: PT show the two coincide when
# there's no serial correlation, so they should roughly agree.
def pesaran_timmermann(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int, float]:
    T = len(actual)
    Y = np.eye(len(CLASSES))[actual][:, 1:]       # drop one class (avoids collinearity)
    Xd = np.eye(len(CLASSES))[predicted][:, 1:]
    Xd = Xd[:, Xd.std(axis=0) > 0]               # a class never predicted carries no information
    if Xd.shape[1] == 0:
        return np.nan, 0, np.nan                  # constant forecast: nothing to test
    Y, Xd = Y - Y.mean(axis=0), Xd - Xd.mean(axis=0)  # demeaning = including an intercept
    XtX_inv = np.linalg.inv(Xd.T @ Xd / T)
    B = XtX_inv @ (Xd.T @ Y / T)                  # slopes, one column per equation
    E = Y - Xd @ B
    g = np.einsum("tj,tk->tjk", E, Xd).reshape(T, -1)  # score per week, ordered as vec(B)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    S = g.T @ g / T
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)                    # Bartlett weights
        G = g[L:].T @ g[:-L] / T
        S += w * (G + G.T)
    A = np.kron(np.eye(Y.shape[1]), XtX_inv)
    V = A @ S @ A / T
    b = B.T.reshape(-1)                           # vec(B), same order as the scores
    stat = float(b @ np.linalg.solve(V, b))
    df = b.size
    return stat, df, 1 - stats.chi2.cdf(stat, df)


pt_rows = []
print("\nPesaran-Timmermann (is the forecast related to the outcome at all?):")
for name in MODELS + ["baseline_last_week"]:
    stat, df, p = pesaran_timmermann(preds["actual"].values, preds[name].values)
    cm = confusion_matrix(preds["actual"], preds[name], labels=range(len(CLASSES)))
    cm = cm[cm.sum(axis=1) > 0][:, cm.sum(axis=0) > 0]
    chi2_p = stats.chi2_contingency(cm)[1] if min(cm.shape) > 1 else np.nan
    pt_rows.append({"forecast": name, "pt_stat": stat, "df": df, "pt_p_value": p, "pearson_chi2_p_value": chi2_p})
    print(f"  {name:20s} PT = {stat:6.2f} (df {df})  p = {p:.3f}   Pearson chi2 p = {chi2_p:.3f}"
          f"  ({'related' if p < 0.05 else 'no evidence of a relation'})")
pd.DataFrame(pt_rows).to_csv(RESULTS_DIR / "pesaran_timmermann.csv", index=False)


# %% 7. Seed robustness
# Random Forest, XGBoost and LightGBM all use randomness (which rows and
# features each tree sees). Rerun with 5 seeds: if the scores move by
# more than the gap between two models, that gap is just noise.
print(f"\nRerunning with seeds {SEEDS} (takes a few minutes)...")
seed_rows = []
for seed in SEEDS:
    p = preds if seed == MAIN_SEED else walk_forward(seed)
    s = score_table(p)
    for name in MODELS:
        seed_rows.append({"seed": seed, "model": name, **s.loc[name].to_dict()})
seed_df = pd.DataFrame(seed_rows)
seed_df.to_csv(RESULTS_DIR / "seed_runs.csv", index=False)
seed_summary = seed_df.groupby("model")[["accuracy", "macro_f1", "brier"]].agg(["mean", "std", "min", "max"])
print("\nAcross seeds:")
print(seed_summary.round(3))
seed_summary.to_csv(RESULTS_DIR / "seed_summary.csv")


# %% 8. Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, name in zip(axes, MODELS):
    cm = confusion_matrix(preds["actual"], preds[name], labels=range(len(CLASSES)))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{name}\nacc {scores.loc[name, 'accuracy']:.2f}, F1 {scores.loc[name, 'macro_f1']:.2f}")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
plt.show()


# %% 9. Accuracy over time
fold_acc = preds.groupby("fold").apply(
    lambda g: pd.Series({n: (g[n] == g["actual"]).mean() for n in ["ensemble", best_baseline]})
)
fold_acc.index = preds.groupby("fold").apply(lambda g: g.index[0])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(fold_acc.index, fold_acc["ensemble"], marker="o", label="ensemble")
ax.plot(fold_acc.index, fold_acc[best_baseline], marker="o", label=best_baseline)
ax.axhline(1 / 3, color="grey", ls="--", lw=1, label="random guess (1/3)")
ax.set_title("Accuracy per 6-month test block")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_over_time.png", dpi=150)
plt.show()


# %% 10. Simple trading check
# Not a full backtest (no costs), just a sense check: long on "buy",
# short on "sell", out on "hold", vs always long.
fwd = data["fwd_return_5d"].reindex(preds.index)
strategy = (preds["ensemble"].map({0: -1, 1: 0, 2: 1}) * fwd).cumsum()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(strategy.index, strategy, label="ensemble signal (long/flat/short)")
ax.plot(fwd.index, fwd.cumsum(), label="always long Brent")
ax.set_title("Cumulative 5-day returns, no transaction costs (sense check only)")
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "strategy_check.png", dpi=150)
plt.show()


# %% 11. Record exactly what was run
# So the write-up (and Models B and C) can show identical settings.
run_info = {
    "model": MODEL_NAME,
    "n_weeks": len(data),
    "n_features": len(feature_cols),
    "features": feature_cols,
    "window_weeks": WINDOW_WEEKS,
    "window_type": "rolling",
    "test_block": TEST_BLOCK,
    "test_period": [str(preds.index[0].date()), str(preds.index[-1].date())],
    "n_test_weeks": len(preds),
    "main_seed": MAIN_SEED,
    "seeds": SEEDS,
    "model_params": {n: m.get_params() for n, m in make_models(MAIN_SEED).items()},
}
(RESULTS_DIR / "run_config.json").write_text(json.dumps(run_info, indent=2, default=str))
print(f"\nEverything saved to {RESULTS_DIR}/")
