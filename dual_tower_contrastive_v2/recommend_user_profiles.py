import os
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_cleaner import auto_clean_data
from utils.preprocess import preprocess_users

def load_model(model_path, user_dim, embed_dim=64, device='cpu'):
    d = torch.load(model_path, map_location=device)
    from models.user_tower import UserTower
    from models.product_tower import ProductTower
    prod_dim = d.get('prod_dim', None)
    prod_cols = d.get('prod_cols', [])
    user_cols = d.get('user_cols', ['gender','age','edu_bg','marriage_situ_cd'])
    if prod_dim is None or prod_dim == 0:
        prod_dim = len(prod_cols) if prod_cols else 1
    user_t = UserTower(len(user_cols), embed_dim)
    prod_t = ProductTower(prod_dim, embed_dim)
    user_t.load_state_dict(d['user_state'])
    prod_t.load_state_dict(d['prod_state'])
    user_t.eval(); prod_t.eval()
    return user_t, prod_t, prod_cols, user_cols

def handle_new_products(new_sub, prod_emb_old, old_prod_ids, prod_t, prod_cols):
    from sklearn.preprocessing import normalize
    results = []
    if new_sub.shape[0] == 0:
        return results
    if len(prod_cols) == 0:
        new_sub['prod_hash'] = new_sub['prod_id'].astype(str).apply(lambda x: hash(x)%1000)
        prod_cols = ['prod_hash']
    prod_matrix_new = new_sub[prod_cols].fillna(0).astype(float).values
    prod_tensor_new = torch.tensor(prod_matrix_new, dtype=torch.float32)
    with torch.no_grad():
        prod_emb_new = prod_t(prod_tensor_new).numpy()
    prod_emb_new = normalize(prod_emb_new, axis=1)
    sims = cosine_similarity(prod_emb_new, prod_emb_old)
    for i, pid in enumerate(new_sub['prod_id']):
        j = int(np.argmax(sims[i]))
        sim_score = float(sims[i][j])
        results.append((pid, old_prod_ids[j], sim_score))
    return results

def main(args):
    cleaned = auto_clean_data(args.data_dir)
    cust_df = cleaned.get('cust_dataset.csv') if cleaned.get('cust_dataset.csv') is not None else pd.read_csv(os.path.join(args.data_dir,'cust_dataset.csv'))
    events_df = cleaned.get('event_dataset.csv') if cleaned.get('event_dataset.csv') is not None else pd.read_csv(os.path.join(args.data_dir,'event_dataset.csv'))
    prod_df = cleaned.get('product_xlsx') if cleaned.get('product_xlsx') is not None else pd.read_excel(os.path.join(args.data_dir,'productLabels_multiSpreadsheets.xlsx'), sheet_name=None)
    if isinstance(prod_df, dict):
        prod_df = pd.concat(list(prod_df.values()), ignore_index=True, sort=False)
    cust_df = preprocess_users(cust_df)
    user_cols = ['gender','age','edu_bg','marriage_situ_cd']
    user_matrix = cust_df[user_cols].fillna(0).astype(float).values
    user_ids = cust_df['cust_no'].astype(str).values if 'cust_no' in cust_df.columns else cust_df.index.astype(str).values
    all_results = []
    cat_store = {}
    for fname in os.listdir(args.model_dir):
        if not fname.startswith('model_') or not fname.endswith('.pth'):
            continue
        cat = fname.split('_')[1].split('.')[0]
        model_path = os.path.join(args.model_dir, fname)
        user_t, prod_t, prod_cols, user_cols_saved = load_model(model_path, user_dim=len(user_cols), embed_dim=args.embed_dim)
        prod_mask = prod_df['prod_id'].astype(str).str.startswith(cat)
        prod_sub = prod_df[prod_mask].reset_index(drop=True)
        if prod_sub.shape[0] == 0:
            continue
        if len(prod_cols) == 0:
            prod_sub['prod_hash'] = prod_sub['prod_id'].astype(str).apply(lambda x: hash(x)%1000)
            prod_cols = ['prod_hash']
        prod_matrix = prod_sub[prod_cols].fillna(0).astype(float).values
        user_tensor = torch.tensor(user_matrix, dtype=torch.float32)
        prod_tensor = torch.tensor(prod_matrix, dtype=torch.float32)
        with torch.no_grad():
            user_emb = user_t(user_tensor).numpy()
            prod_emb = prod_t(prod_tensor).numpy()
        from sklearn.preprocessing import normalize
        user_emb = normalize(user_emb, axis=1)
        prod_emb = normalize(prod_emb, axis=1)
        sims = cosine_similarity(prod_emb, user_emb)
        pop = events_df[events_df['event_type'].isin(['A','B'])].groupby('prod_id').size().to_dict()
        profiles = []
        for i, pid in enumerate(prod_sub['prod_id']):
            topk_idx = np.argsort(sims[i])[-args.k:][::-1]
            topk_users = user_ids[topk_idx].tolist()
            group = cust_df.iloc[topk_idx]
            profile = {
                'prod_id': pid,
                'category': cat,
                'is_new_product': 0,
                'most_similar_existing_prod_id': '',
                'similarity_score': '',
                'gender_male_ratio': float(group['gender'].mean()),
                'age_mean': float(group['age'].mean()),
                'age_median': float(group['age'].median()),
                'edu_mode': str(group['edu_bg'].mode().iloc[0]) if not group['edu_bg'].mode().empty else '',
                'marriage_mode': str(group['marriage_situ_cd'].mode().iloc[0]) if not group['marriage_situ_cd'].mode().empty else '',
                'sample_k': int(len(topk_users)),
                'top_k_custnos': '|'.join(topk_users),
                'popularity': int(pop.get(pid,0))
            }
            profiles.append(profile)
            all_results.append(profile)
        cat_store[cat] = {
            'prod_emb': prod_emb,
            'prod_ids': prod_sub['prod_id'].tolist(),
            'prod_cols': prod_cols,
            'profiles_df': pd.DataFrame(profiles)
        }
    for cat, store in cat_store.items():
        prod_mask = prod_df['prod_id'].astype(str).str.startswith(cat)
        new_sub = prod_df[prod_mask & (prod_df.get('new_flag',0)==1)].reset_index(drop=True)
        new_map = handle_new_products(new_sub, store['prod_emb'], store['prod_ids'], prod_t, store['prod_cols'])
        for pid, old_pid, sim_score in new_map:
            ref = store['profiles_df']
            row = ref[ref['prod_id']==old_pid]
            if row.shape[0] == 0:
                continue
            r = row.iloc[0].to_dict()
            r.update({
                'prod_id': pid,
                'category': cat,
                'is_new_product': 1,
                'most_similar_existing_prod_id': old_pid,
                'similarity_score': float(sim_score),
                'popularity': 0
            })
            all_results.append(r)
    out_df = pd.DataFrame(all_results)
    out_df = out_df.sort_values(['popularity'], ascending=False)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f'[RESULT] Saved recommendations to {args.out_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--model_dir', default='outputs')
    parser.add_argument('--out_csv', default='outputs/recommended_user_profiles.csv')
    parser.add_argument('--k', type=int, default=200)
    parser.add_argument('--embed_dim', type=int, default=64)
    args = parser.parse_args()
    main(args)
