# visualization/plots.py
"""
Generates:
- Feature importance bar
- SHAP summary plot
- Partial dependence for a few top features (if sklearn version supports it)

Saves outputs to OUTPUTS/ (configured in config.py)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import OUTPUTS
from utils.logger import get_logger

logger = get_logger("plots", logfile="plots.log")

# SHAP is optional — try import and gracefully handle missing package
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
    logger.warning("shap not available: to enable SHAP install the 'shap' package")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def explain_model(model, X, model_name="model"):
    outdir = _ensure_dir(OUTPUTS)
    # Feature importance from Booster
    try:
        importance = np.array(model.feature_importance(importance_type="gain"))
    except Exception:
        try:
            importance = np.array(model.feature_importance())
        except Exception:
            importance = np.zeros(X.shape[1])
    features = np.array(X.columns)

    # Sort descending
    idx = np.argsort(importance)[::-1]
    topk = min(30, len(features))
    sorted_features = features[idx][:topk]
    sorted_importance = importance[idx][:topk]

    plt.figure(figsize=(8, max(4, topk*0.25)))
    plt.barh(sorted_features[::-1], sorted_importance[::-1])
    plt.title(f"{model_name} Feature Importance (gain)")
    plt.tight_layout()
    imp_path = os.path.join(outdir, f"{model_name}_feature_importance.png")
    plt.savefig(imp_path, dpi=200)
    plt.close()
    logger.info(f"Saved feature importance to {imp_path}")

    # SHAP
    if _HAS_SHAP:
        try:
            logger.info("Computing SHAP values (may take time)...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            # summary plot
            shap.summary_plot(shap_values, X, show=False)
            shp_path = os.path.join(outdir, f"{model_name}_shap_summary.png")
            plt.tight_layout()
            plt.savefig(shp_path, dpi=200)
            plt.close()
            logger.info(f"Saved SHAP summary to {shp_path}")
        except Exception as e:
            logger.warning(f"SHAP plotting failed: {e}")
    else:
        logger.warning("SHAP not installed; skipping SHAP plots.")
