import os, pandas as pd

def align_product_features(prod_df, min_numeric_cols=4):
    out = {}
    sheets = prod_df['__sheet__'].unique() if '__sheet__' in prod_df.columns else ['ALL']
    for s in sheets:
        sub = prod_df[prod_df.get('__sheet__','ALL')==s].copy()
        drop_cols = ['prod_id','__sheet__','new_flag']
        cols = [c for c in sub.columns if c not in drop_cols]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(sub[c])]
        # factorize some categorical if too few numeric cols
        if len(numeric_cols) < min_numeric_cols:
            for c in cols:
                if c not in numeric_cols:
                    try:
                        sub[c+'_cat'] = pd.factorize(sub[c].astype(str))[0]
                        numeric_cols.append(c+'_cat')
                    except:
                        continue
                if len(numeric_cols) >= min_numeric_cols:
                    break
        if len(numeric_cols) == 0:
            sub['prod_hash'] = sub['prod_id'].astype(str).apply(lambda x: abs(hash(x))%10000)
            numeric_cols = ['prod_hash']
        for c in numeric_cols:
            sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0).astype(float)
        out[s] = {'df': sub, 'numeric_cols': numeric_cols}
    return out
def save_aligned(data_dir, aligned_dict):
    """Save aligned product data per category into data/aligned/"""
    import os
    os.makedirs(os.path.join(data_dir, 'aligned'), exist_ok=True)
    for cat, data in aligned_dict.items():
        df = data['df']
        out_path = os.path.join(data_dir, 'aligned', f'aligned_prod_{cat}.csv')
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
