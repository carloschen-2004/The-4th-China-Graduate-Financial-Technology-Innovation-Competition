import os, re, pandas as pd, numpy as np

CUST_ID_CANDIDATES = ['cust_no','customer_id','custid','客户号','CUST_ID']
PROD_ID_CANDIDATES = ['prod_id','product_id','prod_code','产品编号','PROD_ID','PROD_CD']
EVENT_TYPE_CANDIDATES = ['event_type','trans_type','event_cd','事件类型','交易类型']

def normalize_id_str(s):
    if pd.isna(s):
        return ''
    s = str(s).strip()
    # remove trailing .0 from floats like "1234.0"
    if re.match(r'^\d+\.0$', s):
        s = s[:-2]
    # convert scientific to int if integer
    try:
        if 'e' in s.lower():
            v = float(s)
            if abs(v - int(v)) < 1e-6:
                s = str(int(v))
    except:
        pass
    return s

def find_first_column(df, candidates):
    cols = list(df.columns)
    for c in candidates:
        if c in cols: return c
    # case-insensitive exact match
    lowmap = {col.lower():col for col in cols}
    for c in candidates:
        if c.lower() in lowmap: return lowmap[c.lower()]
    # substring match
    for col in cols:
        for c in candidates:
            if c.lower() in col.lower():
                return col
    return None

def clean_cust_df(path):
    df = pd.read_csv(path, dtype=object)
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # --- 性别处理 ---
    # 标准化列名
    for col in df.columns:
        if col.lower() in ['gender', 'sex', 'gender_cd', 'gender_code', '性别']:
            df.rename(columns={col: 'gender'}, inplace=True)

    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper().map({'M': 1, 'F': 0}).fillna(0.5)
    else:
        df['gender'] = 0.5  # 默认 0.5 表示未知性别

    # --- 年龄处理 ---
    if 'birth_ym' in df.columns:
        def parse_age(x):
            try:
                s = str(x)
                if len(s) >= 4:
                    y = int(s[:4])
                    return 2025 - y
            except:
                return 35
            return 35
        df['age'] = df['birth_ym'].apply(parse_age)
    else:
        df['age'] = 35

    # --- 教育程度 / 婚姻情况 ---
    if 'edu_bg' not in df.columns:
        df['edu_bg'] = 0
    if 'marriage_situ_cd' not in df.columns:
        df['marriage_situ_cd'] = 0

    # 只保留有意义的列
    keep_cols = ['cust_no', 'gender', 'age', 'edu_bg', 'marriage_situ_cd']
    df = df[keep_cols].fillna(0)
    print(f"[CLEAN] cust_df cleaned: {df.shape}, columns={df.columns.tolist()}")
    return df


def clean_event_df(path):
    df = pd.read_csv(path, dtype=object)
    prod_col = find_first_column(df, PROD_ID_CANDIDATES)
    cust_col = find_first_column(df, CUST_ID_CANDIDATES)
    event_col = find_first_column(df, EVENT_TYPE_CANDIDATES)
    if prod_col is None or cust_col is None or event_col is None:
        raise ValueError('prod/cust/event column not found in event dataset')
    df = df.rename(columns={prod_col:'prod_id', cust_col:'cust_no', event_col:'event_type'})
    df['prod_id'] = df['prod_id'].astype(str).apply(normalize_id_str)
    df['cust_no'] = df['cust_no'].astype(str).apply(normalize_id_str)
    df['event_type'] = df['event_type'].astype(str).str.strip()
    # create binary columns for events: A and B treated as positive
    df['A'] = df['event_type'].apply(lambda x: 1 if str(x).upper()=='A' else 0)
    df['B'] = df['event_type'].apply(lambda x: 1 if str(x).upper()=='B' else 0)
    # fill other cols numeric where possible
    for col in df.columns:
        if col in ['prod_id','cust_no','event_type','A','B']: continue
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def clean_product_xlsx(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    raws = []
    for sname, df in sheets.items():
        # ensure prod_id exists, try common names
        if 'prod_id' not in df.columns:
            for c in PROD_ID_CANDIDATES:
                if c in df.columns:
                    df = df.rename(columns={c:'prod_id'})
                    break
        if 'prod_id' not in df.columns:
            raise ValueError(f'prod_id not found in sheet {sname}')
        df['prod_id'] = df['prod_id'].astype(str).apply(normalize_id_str)
        df['__sheet__'] = sname
        # coerce numeric where possible and fillna
        for col in df.columns:
            if col in ['prod_id','__sheet__']: continue
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df[col] = df[col].fillna('').astype(str)
                if df[col].nunique() < 200:
                    df[col] = pd.factorize(df[col].astype(str))[0]
        raws.append(df)
    prod_df = pd.concat(raws, ignore_index=True, sort=False)
    return prod_df

def auto_clean_all(data_dir):
    cleaned_dir = os.path.join(data_dir, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)
    cust = clean_cust_df(os.path.join(data_dir, 'cust_dataset.csv'))
    ev = clean_event_df(os.path.join(data_dir, 'event_dataset.csv'))
    prod = clean_product_xlsx(os.path.join(data_dir, 'productLabels_multiSpreadsheets.xlsx'))
    cust.to_csv(os.path.join(cleaned_dir,'cleaned_cust_dataset.csv'), index=False)
    ev.to_csv(os.path.join(cleaned_dir,'cleaned_event_dataset.csv'), index=False)
    prod.to_excel(os.path.join(cleaned_dir,'cleaned_productLabels_multiSpreadsheets.xlsx'), index=False)
    return cust, ev, prod
