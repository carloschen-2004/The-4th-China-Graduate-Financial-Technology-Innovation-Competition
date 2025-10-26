# recommend_user_profiles.py（替换现有文件）
import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import normalize

def load_model_meta(model_path):
    """Load model file and return saved meta and state dicts."""
    d = torch.load(model_path, map_location='cpu')
    prod_cols = d.get('prod_cols', [])
    user_cols = d.get('user_cols', [])
    prod_dim = d.get('prod_dim', len(prod_cols) if prod_cols else 1)
    return d, prod_cols, user_cols, prod_dim

def ensure_cols(df, cols):
    """Return df[cols] ensuring missing cols are created as zeros."""
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = 0
    return df2[cols]

def safe_mode_value(series):
    if series is None or len(series)==0:
        return ''
    m = series.mode()
    return str(m.iloc[0]) if not m.empty else ''

def compute_user_embeddings(user_t, cust_df, user_cols):
    # Ensure columns exist and numeric
    mat_df = ensure_cols(cust_df, user_cols).fillna(0).astype(float)
    with torch.no_grad():
        emb = user_t(torch.tensor(mat_df.values, dtype=torch.float32)).numpy()
    # normalize
    emb = normalize(emb, axis=1)
    return emb, mat_df

def compute_prod_embeddings(prod_t, prod_df, prod_cols):
    mat_df = ensure_cols(prod_df, prod_cols).fillna(0).astype(float)
    with torch.no_grad():
        emb = prod_t(torch.tensor(mat_df.values, dtype=torch.float32)).numpy()
    emb = normalize(emb, axis=1)
    return emb, mat_df

def fallback_topk_from_events(events_df, prod_id, k):
    # take top customers by count for this product
    sub = events_df[events_df['prod_id']==prod_id]
    if sub.shape[0]==0:
        return []
    counts = sub.groupby('cust_no').size().sort_values(ascending=False)
    return counts.index.astype(str).tolist()[:k]

def main(args):
    # load cleaned if exists
    cleaned_dir = os.path.join(args.data_dir, 'cleaned')
    if os.path.exists(cleaned_dir):
        cust = pd.read_csv(os.path.join(cleaned_dir, 'cleaned_cust_dataset.csv'), dtype=object)
        events = pd.read_csv(os.path.join(cleaned_dir, 'cleaned_event_dataset.csv'), dtype=object)
        # product concatenated
        prod = pd.read_excel(os.path.join(cleaned_dir, 'cleaned_productLabels_multiSpreadsheets.xlsx'), dtype=object)
    else:
        # fallback to raw
        cust = pd.read_csv(os.path.join(args.data_dir, 'cust_dataset.csv'), dtype=object)
        events = pd.read_csv(os.path.join(args.data_dir, 'event_dataset.csv'), dtype=object)
        prod = None
        # Try reading excel if exists
        x = os.path.join(args.data_dir, 'productLabels_multiSpreadsheets.xlsx')
        if os.path.exists(x):
            prod = pd.read_excel(x, sheet_name=None, dtype=object)
            prod = pd.concat(list(prod.values()), ignore_index=True, sort=False)

    # normalize id columns to str
    cust['cust_no'] = cust['cust_no'].astype(str).str.strip()
    events['cust_no'] = events['cust_no'].astype(str).str.strip()
    events['prod_id'] = events['prod_id'].astype(str).str.strip()
    if prod is not None:
        prod['prod_id'] = prod['prod_id'].astype(str).str.strip()

    # Only consider successful events (A or B) as positive interactions
    events['event_type'] = events['event_type'].astype(str).str.upper().fillna('')
    pos_events = events[events['event_type'].isin(['A','B'])].copy()

    # load each model and compute recommendations per model category
    results = []
    model_files = [f for f in os.listdir(args.model_dir) if f.startswith('model_') and f.endswith('.pth')]
    if len(model_files)==0:
        print("[ERR] No model files found in", args.model_dir)
        return

    for mf in model_files:
        cat = mf.split('_',1)[1].rsplit('.',1)[0]
        model_path = os.path.join(args.model_dir, mf)

        # load model meta & states
        d, prod_cols_saved, user_cols_saved, prod_dim = load_model_meta(model_path)
        # import classes dynamically to avoid mismatch issues
        embed_dim_saved = d.get('embed_dim', args.embed_dim)
        user_t = __import__('models.user_tower', fromlist=['UserTower']).UserTower(
            len(user_cols_saved) if user_cols_saved else len([c for c in cust.columns if c != 'cust_no']),
            embed_dim_saved
        )
        prod_t = __import__('models.product_tower', fromlist=['ProductTower']).ProductTower(prod_dim, embed_dim_saved)

        user_t.load_state_dict(d['user_state'])
        prod_t.load_state_dict(d['prod_state'])
        user_t.eval(); prod_t.eval()

        # determine product dataframe rows for this category
        aligned_file = os.path.join(args.data_dir, 'aligned', f'aligned_prod_{cat}.csv')
        if os.path.exists(aligned_file):
            prod_df = pd.read_csv(aligned_file, dtype=object)
        else:
            # fallback to using all prod rows with matching sheet name if available
            if '__sheet__' in prod.columns:
                prod_df = prod[prod['__sheet__']==cat].copy()
            else:
                # take all products if cannot split
                prod_df = prod.copy()

        if prod_df is None or prod_df.shape[0]==0:
            print(f"[WARN] No product rows found for category {cat}, skip")
            continue

        # build user embedding matrix using saved user_cols; fallback to cust-derived
        # ensure cust has columns; create zeros if missing
        if user_cols_saved:
            user_df_for_emb = ensure_cols(cust, user_cols_saved).fillna(0).astype(float)
            user_ids = cust['cust_no'].astype(str).values
        else:
            # fallback: use all cust cols except id
            user_cols_fallback = [c for c in cust.columns if c!='cust_no']
            user_df_for_emb = ensure_cols(cust, user_cols_fallback).fillna(0).astype(float)
            user_ids = cust['cust_no'].astype(str).values

        # compute user embeddings
        with torch.no_grad():
            user_emb = user_t(torch.tensor(user_df_for_emb.values, dtype=torch.float32)).numpy()
        user_emb = normalize(user_emb, axis=1)

        # compute prod embeddings (use saved prod_cols if present)
        prod_cols_use = [c for c in prod_cols_saved if c in prod_df.columns] if prod_cols_saved else []
        if len(prod_cols_use)==0:
            # fallback: choose numeric cols except id
            prod_cols_use = [c for c in prod_df.columns if c not in ['prod_id','__sheet__']][:max(1, len(prod_cols_saved) or 1)]

        prod_mat = ensure_cols(prod_df, prod_cols_use).fillna(0).astype(float)
        with torch.no_grad():
            prod_emb = prod_t(torch.tensor(prod_mat.values, dtype=torch.float32)).numpy()
        prod_emb = normalize(prod_emb, axis=1)

        # compute similarity matrix (prod x users)
        sims = np.matmul(prod_emb, user_emb.T)  # shape (n_prod, n_user)

        # For each product, pick top-k. If sims row is degenerate (all equal or nan), fallback to event-based top purchasers
        for i, pid in enumerate(prod_df['prod_id'].astype(str).values):
            row = sims[i]
            fallback_reason = ''
            # check variability
            if np.all(np.isclose(row, row[0])) or np.isnan(row).all():
                # fallback to event-based top purchasers
                topk_custs = fallback_topk_from_events(pos_events, pid, args.k)
                fallback_reason = 'degenerate_similarity_fallback_to_event'
            else:
                topk_idx = np.argsort(-row)[:args.k]
                topk_custs = user_ids[topk_idx].tolist()

            # if still empty, force empty list
            if not topk_custs:
                topk_custs = []

            # now compute profile stats safely
            topk_df = cust[cust['cust_no'].isin(topk_custs)].copy() if len(topk_custs)>0 else pd.DataFrame(columns=cust.columns)
            gender_male_ratio = None
            age_mean = None
            age_median = None
            edu_mode = ''
            marriage_mode = ''

            if not topk_df.empty:
                if 'gender' in topk_df.columns:
                    # ensure numeric mapping: M->1 F->0 or numeric already
                    g = topk_df['gender'].astype(str).str.upper().map({'M':1,'F':0}).fillna(pd.to_numeric(topk_df['gender'], errors='coerce').fillna(0.0))
                    gender_male_ratio = float(pd.to_numeric(g, errors='coerce').astype(float).mean())
                if 'age' in topk_df.columns:
                    age_mean = float(pd.to_numeric(topk_df['age'], errors='coerce').mean())
                    age_median = float(pd.to_numeric(topk_df['age'], errors='coerce').median())
                if 'edu_bg' in topk_df.columns and not topk_df['edu_bg'].mode().empty:
                    edu_mode = str(topk_df['edu_bg'].mode().iloc[0])
                if 'marriage_situ_cd' in topk_df.columns and not topk_df['marriage_situ_cd'].mode().empty:
                    marriage_mode = str(topk_df['marriage_situ_cd'].mode().iloc[0])

            results_row = {
                'prod_id': pid,
                'category': cat,
                'gender_male_ratio': gender_male_ratio if gender_male_ratio is not None else 0,
                'age_mean': age_mean if age_mean is not None else 0,
                'age_median': age_median if age_median is not None else 0,
                'edu_mode': edu_mode,
                'marriage_mode': marriage_mode,
                'top_k_custnos': '|'.join(topk_custs),
                'sample_k': len(topk_custs),
                'fallback_reason': fallback_reason
            }
            results.append(results_row)

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f"[DONE] saved {args.out_csv} rows={len(out)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--model_dir', default='outputs')
    p.add_argument('--out_csv', default='outputs/recommended_user_profiles.csv')
    p.add_argument('--k', type=int, default=200)
    p.add_argument('--embed_dim', type=int, default=64)
    args = p.parse_args()
    main(args)
