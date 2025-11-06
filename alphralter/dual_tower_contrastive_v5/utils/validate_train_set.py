# utils/validate_train_set.py
import os
import pandas as pd
import numpy as np

def validate_train_set(cust_df, prod_df, events_df, user_emb, prod_emb, user_ids, prod_ids, top_k=200, out_dir='outputs', out_name='modified_train_events.csv'):
    """
    使用现有 embeddings 验证训练集：
    - 对每个用户取 top-K 产品预测
    - 与训练集对比，没有成功事件(A/B)则新增 D 记录
    """
    train_modified = events_df.copy()

    sims = np.matmul(user_emb, prod_emb.T)  # 用户 x 产品

    for i, uid in enumerate(user_ids):
        row = sims[i]
        if np.all(np.isnan(row)) or np.all(np.isclose(row, row[0])):
            topk_idx = []
        else:
            topk_idx = np.argsort(-row)[:top_k]
        topk_prods = prod_ids[topk_idx].tolist() if len(topk_idx) > 0 else []

        for pid in topk_prods:
            exists = train_modified[
                (train_modified['cust_no']==uid) &
                (train_modified['prod_id']==pid) &
                (train_modified['event_type'].isin(['A','B']))
            ]
            if exists.shape[0] == 0:
                # 新增 D 记录
                new_row = {
                    'cust_no': uid,
                    'prod_id': pid,
                    'event_id': '',
                    'event_type': 'D',
                    'event_level': '',
                    'event_date': '',
                    'event_term': '',
                    'event_rate': '',
                    'event_amt': ''
                }
                train_modified = pd.concat([train_modified, pd.DataFrame([new_row])], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    train_modified.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[DONE] saved modified training events to {out_path}, rows={len(train_modified)}")
    return train_modified
