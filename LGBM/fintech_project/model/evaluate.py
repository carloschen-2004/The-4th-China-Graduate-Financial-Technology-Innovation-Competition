# model/evaluate.py
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_score(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")

def precision_at_k_per_user(true_map, pred_map, k=5):
    precisions = []
    for uid, preds in pred_map.items():
        true_set = true_map.get(uid, set())
        if not preds:
            precisions.append(0.0)
            continue
        hits = sum([1 for p in preds[:k] if p in true_set])
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0

def ndcg_at_k_per_user(true_map, pred_map, k=5):
    import math
    ndcgs = []
    for uid, preds in pred_map.items():
        true_set = true_map.get(uid, set())
        dcg = 0.0
        for i,p in enumerate(preds[:k]):
            rel = 1.0 if p in true_set else 0.0
            dcg += (2**rel - 1) / math.log2(i+2)
        ideal = min(len(true_set), k)
        idcg = sum((2**1 - 1) / math.log2(i+2) for i in range(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0
