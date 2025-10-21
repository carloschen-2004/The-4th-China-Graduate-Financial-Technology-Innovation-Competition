# pipeline/multi_step.py
# 确保项目根在 sys.path（如果单独运行此模块）
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# 以下导入必须与 data/ model/ visualization/ 中的函数名一致
from data.data_loader import read_cust, read_events, read_products, build_user_product_pairs
from data.preprocess import merge_for_model
from data.feature_engineering import build_features
from model.train_model import train_lgbm
from model.evaluate import precision_at_k_per_user, ndcg_at_k_per_user
from visualization.plots import explain_model
from utils.logger import get_logger
from config import TEST_SIZE, NEG_SAMPLES_PER_POS, TOPK, RANDOM_SEED, LGB_PARAMS
import os

logger = get_logger("multi_step", logfile="multi_step.log")
np.random.seed(RANDOM_SEED)

def _oof_first_choice_label(df_feat, feature_cols, n_splits=5):
    """
    使用 KFold 在训练集上做 OOF 预测，生成每个 customer 的 first_accepted_oof 标签。
    返回 Series 与 df_feat 索引对齐（0/1）。
    """
    X = df_feat[feature_cols].values
    y = df_feat["is_pos"].values
    oof_scores = np.zeros(len(df_feat))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for tr_idx, vl_idx in kf.split(X):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xvl, yvl = X[vl_idx], y[vl_idx]
        # 训练一个小模型用于 OOF（使用 train_lgbm）
        try:
            bst = train_lgbm(Xtr, ytr, Xvl, yvl, model_name="oof_temp", num_boost_round=100, early_stopping_rounds=10, verbose_eval=0)
            try:
                preds = bst.predict(Xvl, num_iteration=bst.best_iteration)
            except Exception:
                preds = bst.predict(Xvl)
        except Exception:
            preds = np.full(len(vl_idx), ytr.mean())
        oof_scores[vl_idx] = preds
    df_feat = df_feat.copy()
    df_feat["oof_score"] = oof_scores
    # 每个客户按 oof_score 排序取 top1，判断是否命中真实正例集合
    top1 = df_feat.sort_values(["cust_no","oof_score"], ascending=[True,False]).groupby("cust_no").first().reset_index()
    true_map = df_feat[df_feat["is_pos"]==1].groupby("cust_no")["prod_id"].apply(set).to_dict()
    accepted_map = {row["cust_no"]: (1 if row["prod_id"] in true_map.get(row["cust_no"], set()) else 0) for _, row in top1.iterrows()}
    df_feat["first_accepted_oof"] = df_feat["cust_no"].map(accepted_map).fillna(0).astype(int)
    return df_feat["first_accepted_oof"]

def train_and_evaluate(simulate=False, n_customers=None, n_products=None, explain=False, sample_rate=1.0):
    """
    主流程：
    - 加载（或生成）数据
    - 构建 (cust, prod) pairs（正负采样）
    - 特征工程
    - Stage1 模型训练与评估
    - 生成训练集 OOF first_accepted，构建 Stage2（feedback-aware）并训练
    - 生成最终推荐、保存详细和合并格式 CSV
    - （可选）对 Stage2 进行可解释性分析并保存图表
    返回字典包含模型和指标及推荐合并表
    """
    logger.info("Loading data...")
    cust_df = read_cust(sample_rate=sample_rate)
    events_df = read_events(sample_rate=sample_rate)
    prod_df = read_products()

    if n_customers:
        cust_df = cust_df.head(n_customers).copy()
    if n_products:
        prod_df = prod_df.head(n_products).copy()

    logger.info("Building user-product pairs...")
    pairs = build_user_product_pairs(cust_df, events_df, prod_df, neg_per_pos=NEG_SAMPLES_PER_POS)

    logger.info("Merging features...")
    merged = merge_for_model(cust_df, events_df, prod_df, pairs)

    logger.info("Feature engineering...")
    df_feat, feature_cols = build_features(merged, events_df)

    # split by customer to avoid leakage
    unique_cust = df_feat["cust_no"].unique()
    train_cust, val_cust = train_test_split(unique_cust, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_df = df_feat[df_feat["cust_no"].isin(train_cust)].reset_index(drop=True)
    val_df = df_feat[df_feat["cust_no"].isin(val_cust)].reset_index(drop=True)

    X_train = train_df[feature_cols]
    y_train = train_df["is_pos"]
    X_val = val_df[feature_cols]
    y_val = val_df["is_pos"]

    # Stage1
    logger.info("Training Stage1 model...")
    model1 = train_lgbm(X_train, y_train, X_val, y_val, model_name="step1_lgb")

    # predict val
    try:
        val_df["score1"] = model1.predict(X_val, num_iteration=model1.best_iteration)
    except Exception:
        val_df["score1"] = model1.predict(X_val)

    # Stage1 ranking & metrics
    recs_stage1 = val_df.sort_values(["cust_no","score1"], ascending=[True,False]).groupby("cust_no")["prod_id"].apply(list).to_dict()
    true_map = val_df[val_df["is_pos"]==1].groupby("cust_no")["prod_id"].apply(set).to_dict()
    p1 = precision_at_k_per_user(true_map, recs_stage1, k=TOPK)
    n1 = ndcg_at_k_per_user(true_map, recs_stage1, k=TOPK)
    logger.info(f"Stage1 Precision@{TOPK}: {p1:.6f}, NDCG@{TOPK}: {n1:.6f}")

    # Generate OOF-based first_accepted for train_df
    logger.info("Generating OOF first_accepted for training Stage2...")
    train_df = train_df.reset_index(drop=True)
    train_df["first_accepted_oof"] = _oof_first_choice_label(train_df, feature_cols, n_splits=5)

    # For val, simulate first_accepted using stage1 predictions (oracle)
    val_users = val_df["cust_no"].unique().tolist()
    first_choice = {u: (recs_stage1.get(u, [None])[0]) for u in val_users}
    accepted_map = {u: (1 if first_choice.get(u) in true_map.get(u, set()) else 0) for u in val_users}
    val_df["first_accepted"] = val_df["cust_no"].map(accepted_map).fillna(0).astype(int)

    # Prepare train/val for Stage2
    train_df["first_accepted"] = train_df["first_accepted_oof"].fillna(0).astype(int)
    feat2 = feature_cols + ["first_accepted"]

    logger.info("Training Stage2 model (feedback-aware)...")
    model2 = train_lgbm(train_df[feat2], train_df["is_pos"], val_df[feat2], val_df["is_pos"], model_name="step2_lgb")

    try:
        val_df["score2"] = model2.predict(val_df[feat2], num_iteration=model2.best_iteration)
    except Exception:
        val_df["score2"] = model2.predict(val_df[feat2])

    # Build final recommendations (per customer top-K)
    final_recs = {}
    for uid, group in val_df.groupby("cust_no"):
        acc = accepted_map.get(uid, 0)
        ordered = group.sort_values("score2", ascending=False)["prod_id"].tolist()
        if acc:
            final_recs[uid] = [p for p in ordered if p != first_choice.get(uid)][:TOPK]
        else:
            final_recs[uid] = ordered[:TOPK]

    # Build detailed recommendations DataFrame (one row per recommended product)
    rec_rows = []
    for uid, prods in final_recs.items():
        for rank, pid in enumerate(prods, start=1):
            # try get score2 value (may be missing if prod not in val_df for that uid)
            score_vals = val_df[(val_df["cust_no"]==uid) & (val_df["prod_id"]==pid)]["score2"]
            score_val = float(score_vals.values[0]) if not score_vals.empty else None
            rec_rows.append({"cust_no": uid, "rank": rank, "prod_id": pid, "score2": score_val})
    recommendations = pd.DataFrame(rec_rows)

    # Save detailed recommendations
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    detailed_path = os.path.join(out_dir, "final_recommendations.csv")
    recommendations.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved detailed recommendations to {detailed_path}")

    # Merge per customer into single row (prod list ordered by score desc)
    if not recommendations.empty:
        recommendations_sorted = recommendations.sort_values(by=["cust_no", "score2"], ascending=[True, False])
        merged_recs = (
            recommendations_sorted.groupby("cust_no")["prod_id"]
            .apply(lambda x: ",".join([str(i) for i in x]))
            .reset_index()
            .rename(columns={"prod_id": "recommended_products"})
        )
    else:
        merged_recs = pd.DataFrame(columns=["cust_no", "recommended_products"])

    merged_path = os.path.join(out_dir, "final_recommendations_merged.csv")
    merged_recs.to_csv(merged_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved merged recommendations to {merged_path}")

    # Compute final metrics from Stage2 ranking
    recs_final_map = recommendations.sort_values(["cust_no","rank"]).groupby("cust_no")["prod_id"].apply(list).to_dict()
    p_final = precision_at_k_per_user(true_map, recs_final_map, k=TOPK)
    n_final = ndcg_at_k_per_user(true_map, recs_final_map, k=TOPK)

    # user-level hit rates
    step1_hits = [1 if first_choice.get(u) in true_map.get(u, set()) else 0 for u in val_users]
    step1_hitrate = np.mean(step1_hits) if len(step1_hits)>0 else 0.0
    combined_hits = []
    for u in val_users:
        if accepted_map.get(u, 0) == 1:
            combined_hits.append(1)
        else:
            combined_hits.append(1 if len(set(final_recs.get(u, [])).intersection(true_map.get(u, set()))) > 0 else 0)
    combined_hitrate = np.mean(combined_hits) if len(combined_hits)>0 else 0.0
    improvement = (combined_hitrate - step1_hitrate) / (step1_hitrate + 1e-9) if step1_hitrate > 0 else float("inf")

    metrics = {
        "stage1_precision_atk": p1,
        "stage1_ndcg_atk": n1,
        "final_precision_atk": p_final,
        "final_ndcg_atk": n_final,
        "stage1_user_hitrate_top1": step1_hitrate,
        "combined_user_hitrate": combined_hitrate,
        "relative_improvement": improvement
    }

    logger.info(f"Metrics: {metrics}")

    # explain (SHAP etc.) if requested
    if explain:
        logger.info("Generating explainability outputs for Stage2...")
        X_val_feat = val_df[feat2]
        try:
            explain_model(model2, X_val_feat, model_name="stage2_lgb")
        except Exception as e:
            logger.warning(f"explain_model failed: {e}")

    return {
        "model1": model1,
        "model2": model2,
        "metrics": metrics,
        "recommendations_detailed": recommendations,
        "recommendations_merged": merged_recs
    }
