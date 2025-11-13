import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import normalize
from utils import accuracy_test

# work_dir: TowTowerNew
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(current_script_path)


def load_model_meta(model_path):
    """磁盘加载模型保存字典，映射到 CPU
    -> <state_dict>,<prod_cols>,<user_cols>,<prod_dim>"""
    d = torch.load(model_path, map_location='cpu') # 模型的 state_dict
    prod_cols = d.get('prod_cols',[])
    user_cols = d.get('user_cols',[])
    prod_dim = d.get('prod_dim', len(prod_cols) if prod_cols else 1)
    return d, prod_cols, user_cols, prod_dim

def ensure_cols(df, cols):
    """确保DataFrame中存在指定列，若缺失则自动添加并填充为0"""
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = 0
    return df2[cols]

def safe_mode_value(series):
    """安全地获取一个 pandas Series 的众数，返回字符串"""
    if series is None or len(series) == 0:
        return ''
    m = series.mode()
    return str(m.iloc[0]) if not m.empty else ''

def compute_user_embedding(user_t, merged_df, user_cols):
    """计算 user_cols 的 embedding"""
    mat_df = ensure_cols(merged_df, user_cols).fillna(0).astype(float)
    with torch.no_grad():
        emb = user_t(torch.tensor(mat_df.values, dtype=torch.float32)).numpy()
    # 标准化
    emb = normalize(emb, axis=1)
    return emb, mat_df

def compute_prod_embeddings(prod_t, merged_df, prod_cols):
    """计算 prod_cols 的 embedding"""
    mat_df = ensure_cols(merged_df, prod_cols).fillna(0).astype(float)
    with torch.no_grad():
        emb = prod_t(torch.tensor(mat_df.values, dtype=torch.float32)).numpy()
    emb = normalize(emb, axis=1)
    return emb, mat_df

def fallback_topk_from_events(event_df, prod_id, k):
    """失败时召回"""
    sub = event_df[event_df["prod_id"] == prod_id]
    if sub.shape[0] == 0:
        return []
    counts = sub.groupby('cust_no').size().sort_values(ascending=False)
    return counts.index.astype(str).tolist()[:k]



#===============main================#
def main(args):
    # load cleaned if exists
    cleaned_dir  = os.path.join(work_dir, 'cleaned')
    events = pd.read_csv(os.path.join(cleaned_dir, 'cleaned_event_dataset.csv'), dtype=object)
    cats = list(events['prod_cat'].unique())
    cats = [str(c) for c in cats]
    levels = events['event_level'].unique()
    print('cats:',cats)
    events['cust_no'] = events['cust_no'].astype(str).str.strip()
    events['prod_id'] = events['prod_id'].astype(str).str.strip()

    # 仅考虑成功事件
    events['event_type'] = events['event_type'].astype(str).str.upper().fillna('')
    # 加载模型，计算每个模型类别下的推荐list
    results = []
    model_files = [f for f in os.listdir(args.model_dir) if f.startswith('model_') and f.endswith('.pth')]

    recommend_custs_df = pd.DataFrame() # 建立空的推荐客户的数据框

    for mf in model_files:
        cat = mf.split('_',1)[1].rsplit('.',1)[0]
        model_path = os.path.join(args.model_dir, mf)

        # 加载元数据
        d, prod_cols_saved, user_cols_saved, prod_dim = load_model_meta(model_path)
        embed_dim_saved = d.get('embed_dim',args.embed_dim)
        user_t = __import__('models.user_tower', fromlist=['UserTower']).UserTower(
            len(user_cols_saved), embed_dim_saved)
        prod_t = __import__('models.product_tower', fromlist=['ProductTower']).ProductTower(
            prod_dim, embed_dim_saved)

        user_t.load_state_dict(d['user_state'])
        prod_t.load_state_dict(d['prod_state'])
        user_t.eval()
        prod_t.eval()

        # 为本类筛选
        aligned_file = os.path.join(work_dir, 'aligned', f'aligned_prod_{cat}.csv')
        merged_df = pd.read_csv(os.path.join(work_dir, f'merged/merged_of_Prod{cat}_test.csv'), dtype=object)
        merged_df = merged_df.drop_duplicates()
        prod_df = pd.read_csv(aligned_file, dtype=object)
        cust_feats = ['gender','age','edu_bg','marriage_situ_cd']
        pos_events = merged_df[merged_df['event_type'].isin(['A', 'B'])].copy()

        # build user embedding matrix using saved user_cols
        user_df_for_emb = ensure_cols(merged_df, user_cols_saved).fillna(0).astype(float)
        ids_hist_list = (['cust_no']+
                         [f'prev_count_{c}' for c in cats]+
                         [f'prev_count_{c}_neg' for c in cats]+cust_feats
                         + [f'event_level_{level}' for level in levels]
                         + ['event_term', 'event_rate', 'event_amt'])
        user_ids_hist = merged_df[ids_hist_list]

        # compute user embeddings
        with torch.no_grad():
            user_emb = user_t(torch.tensor(user_df_for_emb.values, dtype=torch.float32)).numpy()
        user_emb = normalize(user_emb, axis=1)
        # compute prod embeddings
        prod_cols_use = [c for c in prod_cols_saved if c in prod_df.columns]
        prod_mat = ensure_cols(prod_df, prod_cols_use).fillna(0).astype(float)
        with torch.no_grad():
            prod_emb = prod_t(torch.tensor(prod_mat.values, dtype=torch.float32)).numpy()
        prod_emb = normalize(prod_emb, axis=1)

        # compute similarity matrix (prod x users)
        sims = np.matmul(prod_emb, user_emb.T)  # shape (n_prod, n_user)

        # For each product, pick top-k
        # If sims row is degenerate (all equal or nan), fallback to event-based top purchasers
        for i, pid in enumerate(prod_df['prod_id'].astype(str).values):
            row = sims[i]
            fallback_reason = ''
            # check variability
            if np.all(np.isclose(row, row[0])) or np.isnan(row).all():
                # fallback to event-based top purchasers
                topk_custs = fallback_topk_from_events(pos_events,pid,args.k)
                fallback_reason = 'degenerate_similarity_fallback_to_event'
            else:
                topk_idx = np.argsort(-row)[:args.k] # 返回排序后元素在原数组中的索引
                topk_custs = user_ids_hist.iloc[topk_idx].to_dict('records') # 字典化
                recommend_custs = user_ids_hist.iloc[topk_idx] # 准备输出推荐的客户
                recommend_custs = recommend_custs.copy()
                recommend_custs.loc[:,"prod_id"] = pid

                recommend_custs_df = pd.concat([recommend_custs_df,recommend_custs],ignore_index=True)

            if not topk_custs:
                topk_custs = []

            # 计算用户画像
            topk_custs_df = pd.DataFrame(topk_custs)
            if len(topk_custs) > 0:
                topk_df = user_ids_hist.merge(topk_custs_df.drop_duplicates(), how='inner')
            else:
                topk_df = pd.DataFrame(columns=user_ids_hist.columns)
            gender_male_ratio = None
            age_mean = None
            age_median = None
            edu_mode = ''
            marriage_mode = ''

            if not topk_df.empty:
                if 'gender' in topk_df.columns:
                    # ensure numeric mapping: M->1 F->0 or numeric already
                    g = topk_df['gender'].astype(str).str.upper().map({'M': 1, 'F': 0}).fillna(
                        pd.to_numeric(topk_df['gender'], errors='coerce').fillna(0.0))
                    gender_male_ratio = float(pd.to_numeric(g, errors='coerce').astype(float).mean())
                if 'age' in topk_df.columns:
                    age_mean = float(pd.to_numeric(topk_df['age'], errors='coerce').mean())
                    age_median = float(pd.to_numeric(topk_df['age'], errors='coerce').median())
                if 'edu_bg' in topk_df.columns and not topk_df['edu_bg'].mode().empty:
                    edu_mode = str(topk_df['edu_bg'].mode().iloc[0])
                if 'marriage_situ_cd' in topk_df.columns and not topk_df['marriage_situ_cd'].mode().empty:
                    marriage_mode = str(topk_df['marriage_situ_cd'].mode().iloc[0])

                topk_custs_df = topk_custs_df.drop(columns=cust_feats)
                topk_custs = topk_custs_df.to_dict('records')
            results_row = {
                'prod_id': pid,
                'category': cat,
                'gender_male_ratio': gender_male_ratio if gender_male_ratio is not None else 0,
                'age_mean': age_mean if age_mean is not None else 0,
                'age_median': age_median if age_median is not None else 0,
                'edu_mode': edu_mode,
                'marriage_mode': marriage_mode,
                'top_k_custnos': '\n'.join([str(topk_cust) for topk_cust in topk_custs]),
                'sample_k': len(topk_custs),
                'fallback_reason': fallback_reason
            }
            results.append(results_row)

    # 导出推荐文档
    recommend_custs_df['rec_cat'] = recommend_custs_df['prod_id'].apply(lambda v: str(v).strip()[0])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    recommend_custs_df.to_csv(args.out_csv)

    print(f"[DONE] saved {args.out_csv} rows={len(recommend_custs_df)}")

    out_df = pd.DataFrame(results)
    # 除了prod_id以外完全相同的列合并
    group_cols = [col for col in out_df.columns if col != 'prod_id']
    out = out_df.groupby(group_cols)['prod_id'].agg(list).reset_index()
    # 调整输出顺序
    out = out.sort_values(by='prod_id', key=lambda x: x.apply(lambda lst: lst[0]))
    cols = ['prod_id'] + group_cols
    out = out[cols]

    # 计算准确率
    acc_1_list = []
    acc_2_list = []
    for _, row in out.iterrows():
        cust_list = row['top_k_custnos']
        prod_type = row['category']
        prod_id_list = row['prod_id']
        acc_1, acc_2 = accuracy_test.compute_accuracy(cust_list, prod_type, prod_id_list)
        acc_1_list.append(acc_1)
        acc_2_list.append(acc_2)
    out['acc_1'] = acc_1_list
    out['acc_2'] = acc_2_list

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    out.to_excel(args.out_xlsx, index=False, engine='openpyxl')

    print(f"[DONE] saved {args.out_xlsx} rows={len(out)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--model_dir', default='outputs')
    p.add_argument('--out_csv', default='outputs/recommended_custs.csv')
    p.add_argument('--out_xlsx', default='outputs/recommended_user_profiles.xlsx')
    p.add_argument('--k', type=int, default=200)
    p.add_argument('--embed_dim', type=int, default=64)
    args = p.parse_args()
    main(args)



