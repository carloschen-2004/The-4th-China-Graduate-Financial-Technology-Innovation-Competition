import os
import pandas as pd
from openpyxl import load_workbook
import numpy as np

def clean_dataframe(df, name=None):
    # drop fully empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
    # replace empty strings with NaN then fill
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    # fill numeric and object NaNs with 0
    df = df.fillna(0)
    # try to coerce numeric columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass
    return df

def auto_clean_data(data_dir):
    """Return cleaned dataframes without overwriting originals.
    Saves cleaned copies to data/cleaned/ but returns DataFrames for direct use.
    """
    cleaned_dir = os.path.join(data_dir, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)
    results = {}
    # CSV files
    csvs = ['cust_dataset.csv', 'event_dataset.csv']
    for fn in csvs:
        p = os.path.join(data_dir, fn)
        if not os.path.exists(p):
            print(f'[CLEAN] Missing {fn} in {data_dir}; skipping.')
            continue
        df = pd.read_csv(p, dtype=object)
        df = clean_dataframe(df, name=fn)
        if 'cust_no' in df.columns:
            df['cust_no'] = df['cust_no'].astype(str)
        if 'prod_id' in df.columns:
            df['prod_id'] = df['prod_id'].astype(str)
        outp = os.path.join(cleaned_dir, 'cleaned_' + fn)
        df.to_csv(outp, index=False)
        results[fn] = df
        print(f'[CLEAN] {fn}: shape={df.shape} saved to {outp}')
    # Excel products multi-sheet
    xlsx = os.path.join(data_dir, 'productLabels_multiSpreadsheets.xlsx')
    if os.path.exists(xlsx):
        xls = pd.read_excel(xlsx, sheet_name=None, dtype=object)
        cleaned_sheets = {}
        for sname, df in xls.items():
            df = clean_dataframe(df, name=f'sheet:{sname}')
            if 'prod_id' in df.columns:
                df['prod_id'] = df['prod_id'].astype(str)
            df['__sheet__'] = sname
            cleaned_sheets[sname] = df
            print(f'[CLEAN] product sheet {sname}: shape={df.shape}')
        prod_df = pd.concat(list(cleaned_sheets.values()), ignore_index=True, sort=False)
        outp = os.path.join(cleaned_dir, 'cleaned_productLabels_multiSpreadsheets.xlsx')
        with pd.ExcelWriter(outp, engine='openpyxl') as writer:
            for sname, df in cleaned_sheets.items():
                df.to_excel(writer, sheet_name=sname, index=False)
        results['product_xlsx'] = prod_df
        print(f'[CLEAN] product xlsx concatenated shape={prod_df.shape} saved to {outp}')
    else:
        print('[CLEAN] productLabels_multiSpreadsheets.xlsx not found; skipping.')
    return results
