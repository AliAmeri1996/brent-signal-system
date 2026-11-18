"""
config.py

The FROZEN settings shared by Models A, B and C.

Every model script imports from here, so A, B and C are guaranteed to
use the same target, window, test weeks, seeds and model settings. The
only thing allowed to differ between A, B and C is the feature list.
That is what makes the comparison fair.

Don't change anything here after looking at test results (that would be
data snooping). If a change is ever needed, rerun A, B and C together.
"""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Target (used by 06_build_dataset.py) ---------------------------------
THRESHOLD = 0.015   # +/-1.5% over the next 5 trading days -> buy / sell, else hold
HORIZON = 5         # trading days

CLASSES = ["sell", "hold", "buy"]

# --- Walk-forward design ---------------------------------------------------
# ROLLING window of fixed length (your plan, section on Giacomini-White:
# their test needs an estimation window of fixed length). Each fold
# trains on the most recent 260 weeks (~5 years) only, then predicts the
# next 26 weeks (~6 months).
WINDOW_WEEKS = 260
TEST_BLOCK = 26

# --- Randomness ------------------------------------------------------------
# MAIN_SEED gives the headline results. The other seeds are only used to
# show how much the scores move from randomness alone.
MAIN_SEED = 42
SEEDS = [42, 7, 123, 2024, 31337]


# --- Models ----------------------------------------------------------------
# Fixed, sensible settings for a small, noisy dataset (shallow trees,
# many of them). Not tuned on test data.
#
# Class weighting ("balanced") makes each class count equally in
# training, so the models don't lean towards whichever class happened to
# be most common in the training window. This was planned in advance
# ("handling class imbalance in the hold class"), so it isn't tuning.
# XGBoost has no class_weight setting, so the training script gives it
# balanced sample weights instead.
def make_models(seed: int) -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=seed, n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, num_leaves=15, max_depth=4, learning_rate=0.05,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=20, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbose=-1,
        ),
    }
