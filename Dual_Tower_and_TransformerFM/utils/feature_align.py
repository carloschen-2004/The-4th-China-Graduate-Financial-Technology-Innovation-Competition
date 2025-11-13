import pandas as pd
import os

# work_dir: TowTowerNew
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(os.path.dirname(current_script_path))

def align_product_features(prod_df, min_numeric_cols=4):
    out = {}
    sheets = prod_df['__sheet__'].unique()
    for s in sheets:
        sub = prod_df[prod_df['__sheet__'] == s].copy()
        drop_cols = ['prod_id','__sheet__','new_flag']
        cols = [c for c in sub.columns if c not in drop_cols]
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(sub[c])]
        # factorize some categorical if too few numeric cols
        if len(numeric_cols) < min_numeric_cols:
            for c in cols:
                if c not in numeric_cols:
                    try:
                        sub[c + '_cat'] = pd.factorize(sub[c].astype(str))[0]
                        numeric_cols.append(c + '_cat')
                    except:
                        continue
                if len(numeric_cols) >= min_numeric_cols:
                    break
        if len(numeric_cols) == 0:
            sub['prod_hash'] = sub['prod_id'].astype(str).apply(lambda x: abs(hash(x)) % 10000)
            numeric_cols = ['prod_hash']
        for c in numeric_cols:
            sub[c] = pd.to_numeric(sub[c], errors='coerce').fillna(0).astype(float)
        out[s] = {'df': sub, 'numeric_cols': numeric_cols}
    return out

def save_aligned(aligned_dict):
    """Save aligned product data per category into ./aligned/"""
    os.makedirs(os.path.join(work_dir, 'aligned'), exist_ok=True)
    for cat, data in aligned_dict.items():
        df = data['df']
        out_path = os.path.join(work_dir, 'aligned', f'aligned_prod_{cat}.csv')
        df.to_csv(out_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    cleaned_dir = os.path.join(work_dir, 'cleaned')
    prod_df = pd.read_excel(os.path.join(cleaned_dir, 'cleaned_productLabels_multiSpreadsheets.xlsx'))
    aligned_dict = align_product_features(prod_df)
    save_aligned(aligned_dict=aligned_dict)
    print("[FEATURE ALIGN] All categories done and saved in aligned")