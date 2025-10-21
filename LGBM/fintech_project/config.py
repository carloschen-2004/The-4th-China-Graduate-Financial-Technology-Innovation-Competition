# config.py
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS = os.path.join(PROJECT_ROOT, "models")
LOGS = os.path.join(PROJECT_ROOT, "logs")
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")

for d in [DATA_RAW, DATA_PROCESSED, MODELS, LOGS, OUTPUTS]:
    os.makedirs(d, exist_ok=True)

# Modeling params
RANDOM_SEED = 42
TEST_SIZE = 0.2
NEG_SAMPLES_PER_POS = 50
TOPK = 5
SAMPLE_RATE = 1.0

# LightGBM defaults (can be overridden)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": RANDOM_SEED,
}
