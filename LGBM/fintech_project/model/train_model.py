# model/train_model.py
import os
import joblib
import lightgbm as lgb
from config import MODELS, LGB_PARAMS
from utils.logger import get_logger

logger = get_logger("train_model", logfile="train_model.log")

def train_lgbm(X_train, y_train, X_val, y_val, model_name="lgb_model", params=None, num_boost_round=500, early_stopping_rounds=30, verbose_eval=100):
    """
    Train LightGBM booster and save model.
    Uses callbacks for early stopping and log evaluation (compatible with LGBM versions).
    Returns trained Booster.
    """
    os.makedirs(MODELS, exist_ok=True)
    params = params or LGB_PARAMS.copy()

    try:
        pos = max(1, int(y_train.sum()))
        neg = max(1, len(y_train) - pos)
        params["scale_pos_weight"] = neg / pos
    except Exception:
        pass

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = []
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    if verbose_eval and verbose_eval > 0:
        callbacks.append(lgb.log_evaluation(period=verbose_eval))

    logger.info("Starting LightGBM training...")
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=callbacks
    )

    # Save model text and joblib
    model_txt = os.path.join(MODELS, f"{model_name}.txt")
    model_pkl = os.path.join(MODELS, f"{model_name}.pkl")
    try:
        bst.save_model(model_txt)
    except Exception as e:
        logger.warning(f"bst.save_model failed: {e}")
    try:
        joblib.dump(bst, model_pkl)
    except Exception as e:
        logger.warning(f"joblib.dump(bst) failed (Booster may not be picklable in some envs): {e}")

    logger.info(f"Saved model: {model_txt} (and attempted pickle)")
    return bst
