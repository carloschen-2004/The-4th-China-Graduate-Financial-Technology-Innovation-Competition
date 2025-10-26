import os
import re
import pandas as pd
import numpy as np

# ------------------ ID & 类型候选列 ------------------
CUST_ID_CANDIDATES = ['cust_no', 'customer_id', 'custid', '客户号', 'CUST_ID']
PROD_ID_CANDIDATES = ['prod_id', 'product_id', 'prod_code', '产品编号', 'PROD_ID', 'PROD_CD']
EVENT_TYPE_CANDIDATES = ['event_type', 'trans_type', 'event_cd', '事件类型', '交易类型']


# ------------------ 工具函数 ------------------
def normalize_id_str(s):
    """标准化 ID 字符串"""
    if pd.isna(s):
        return ''
    s = str(s).strip()
    # 去掉 float 尾部 .0
    if re.match(r'^\d+\.0$', s):
        s = s[:-2]
    # 科学计数法转整数
    try:
        if 'e' in s.lower():
            v = float(s)
            if abs(v - int(v)) < 1e-6:
                s = str(int(v))
    except:
        pass
    return s


def find_first_column(df, candidates):
    """在 df 中找到第一个匹配候选列"""
    cols = list(df.columns)
    # 精确匹配
    for c in candidates:
        if c in cols: return c
    # 不区分大小写匹配
    lowmap = {col.lower(): col for col in cols}
    for c in candidates:
        if c.lower() in lowmap: return lowmap[c.lower()]
    # 子串匹配
    for col in cols:
        for c in candidates:
            if c.lower() in col.lower():
                return col
    return None


# ------------------ 客户表清洗 ------------------
def clean_cust_df(path):
    df = pd.read_csv(path, dtype=object)
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # 性别处理
    for col in df.columns:
        if col.lower() in ['gender', 'sex', 'gender_cd', 'gender_code', '性别']:
            df.rename(columns={col: 'gender'}, inplace=True)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper().map({'M': 1, 'F': 0}).fillna(0.5)
    else:
        df['gender'] = 0.5

    # 年龄处理
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

    # 教育程度 / 婚姻情况
    if 'edu_bg' not in df.columns:
        df['edu_bg'] = 0
    if 'marriage_situ_cd' not in df.columns:
        df['marriage_situ_cd'] = 0

    # 保留必要列
    keep_cols = ['cust_no', 'gender', 'age', 'edu_bg', 'marriage_situ_cd']
    df = df[keep_cols].fillna(0)

    print(f"[CLEAN] cust_df cleaned: {df.shape}, columns={df.columns.tolist()}")
    return df


# ------------------ 事件表清洗 ------------------
def clean_event_df(path):
    """清洗事件表，保留 event_level 和 event_date 原始值"""
    df = pd.read_csv(path, dtype=object)

    # 找关键列
    prod_col = find_first_column(df, PROD_ID_CANDIDATES)
    cust_col = find_first_column(df, CUST_ID_CANDIDATES)
    event_col = find_first_column(df, EVENT_TYPE_CANDIDATES)
    if prod_col is None or cust_col is None or event_col is None:
        raise ValueError('prod/cust/event column not found in event dataset')

    # 重命名
    df = df.rename(columns={prod_col: 'prod_id', cust_col: 'cust_no', event_col: 'event_type'})

    # 标准化 ID
    df['prod_id'] = df['prod_id'].astype(str).apply(normalize_id_str)
    df['cust_no'] = df['cust_no'].astype(str).apply(normalize_id_str)
    df['event_type'] = df['event_type'].astype(str).str.strip()

    # A/B 二值化
    df['A'] = df['event_type'].apply(lambda x: 1 if str(x).upper() == 'A' else 0)
    df['B'] = df['event_type'].apply(lambda x: 1 if str(x).upper() == 'B' else 0)

    # 保留 event_level 和 event_date 原值
    if 'event_level' in df.columns:
        df['event_level'] = df['event_level'].fillna('unknown')
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

    # 数值化其他列
    for col in df.columns:
        if col in ['prod_id', 'cust_no', 'event_type', 'A', 'B', 'event_level', 'event_date']:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print(f"[CLEAN] event_df cleaned: {df.shape}, columns={df.columns.tolist()}")
    return df


# ------------------ 产品表清洗 ------------------
def clean_product_xlsx(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    raws = []
    for sname, df in sheets.items():
        # 确保 prod_id 存在
        if 'prod_id' not in df.columns:
            for c in PROD_ID_CANDIDATES:
                if c in df.columns:
                    df = df.rename(columns={c: 'prod_id'})
                    break
        if 'prod_id' not in df.columns:
            raise ValueError(f'prod_id not found in sheet {sname}')
        df['prod_id'] = df['prod_id'].astype(str).apply(normalize_id_str)
        df['__sheet__'] = sname
        # 数值化其他列
        for col in df.columns:
            if col in ['prod_id', '__sheet__']:
                continue
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df[col] = df[col].fillna('').astype(str)
                if df[col].nunique() < 200:
                    df[col] = pd.factorize(df[col].astype(str))[0]
        raws.append(df)
    prod_df = pd.concat(raws, ignore_index=True, sort=False)
    print(f"[CLEAN] product_df cleaned: {prod_df.shape}, sheets={list(sheets.keys())}")
    return prod_df


# ------------------ 自动全量清洗 ------------------
def auto_clean_all(data_dir):
    """全量清洗函数，输出 cleaned 文件夹"""
    cleaned_dir = os.path.join(data_dir, 'cleaned')
    os.makedirs(cleaned_dir, exist_ok=True)

    # 清洗客户表
    cust = clean_cust_df(os.path.join(data_dir, 'cust_dataset.csv'))
    cust.to_csv(os.path.join(cleaned_dir, 'cleaned_cust_dataset.csv'), index=False)

    # 清洗事件表
    ev = clean_event_df(os.path.join(data_dir, 'event_dataset.csv'))
    ev.to_csv(os.path.join(cleaned_dir, 'cleaned_event_dataset.csv'), index=False)

    # 清洗产品表
    prod = clean_product_xlsx(os.path.join(data_dir, 'productLabels_multiSpreadsheets.xlsx'))
    prod.to_excel(os.path.join(cleaned_dir, 'cleaned_productLabels_multiSpreadsheets.xlsx'), index=False)

    print(f"[AUTO CLEAN] All datasets cleaned and saved in {cleaned_dir}")
    return cust, ev, prod
