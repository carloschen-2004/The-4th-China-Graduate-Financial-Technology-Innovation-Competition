import os, re, pandas as pd, numpy as np

CUST_ID_CANDIDATES = ['cust_no','customer_id','custid','客户号','CUST_ID']
PROD_ID_CANDIDATES = ['prod_id','product_id','prod_code','产品编号','PROD_ID','PROD_CD']
EVENT_TYPE_CANDIDATES = ['event_type','trans_type','event_cd','事件类型','交易类型']

def normalize_id_str(id_str):
    """标准化ID字符串"""
    if pd.isna(id_str):
        return ''
    id_str = id_str.strip()
    # 去除float尾部.0
    if re.match(r'^\d+\.0$', id_str):
        id_str = id_str[:-2]
    # 科学计数法转整数
    try:
        if 'e' in id_str.lower():
            v = float(id_str)
            if abs(v-int(v)) < 1e-6:
                id_str = str(int(v))
    except:
        pass
    return id_str

def compute_previous_event_counts_with_success(df):
    """计算历史事件类别计数，并区分成功(pos)和失败(neg)"""
    df_processed = df.copy()
    df_processed['_original_sort_order'] = df_processed.reset_index().index
    unique_prod_cats = sorted(df_processed['prod_cat'].unique())
    print(f"所有唯一的产品类别 (Unique prod_cat): {unique_prod_cats}")
    # 步骤 1: 独热编码
    df_dummies = pd.get_dummies(df_processed, columns=['prod_cat'], prefix='count', prefix_sep='_')
    dummy_cols = [f'count_{cat}' for cat in unique_prod_cats]
    # 步骤 2: 基于 is_success 分离成功和失败计数
    # 创建新的列，例如: count_A_pos, count_A_neg, count_C_pos, ...
    outcome_cols = []  # 存储所有新结果列的名称，如 ['count_A_pos', 'count_A_neg', ...]
    rename_map = {}  # 存储最终的重命名映射
    for cat in unique_prod_cats:
        dummy_col = f'count_{cat}'
        # 成功的中间列名
        pos_col_name = f'count_{cat}_pos'
        # 失败的中间列名
        neg_col_name = f'count_{cat}_neg'
        # 计算成功： 只有当事件是该类别 (dummy_col=1) 且 成功 (is_success=1) 时，才记为1
        df_dummies[pos_col_name] = df_dummies[dummy_col] * df_dummies['is_success']
        # 计算失败： 只有当事件是该类别 (dummy_col=1) 且 失败 (is_success=0) 时，才记为1
        # (1 - is_success) 会将 0 变为 1，将 1 变为 0
        df_dummies[neg_col_name] = df_dummies[dummy_col] * (1 - df_dummies['is_success'])
        # 添加到列表，供后续 groupby 使用
        outcome_cols.extend([pos_col_name, neg_col_name])
        # 准备重命名映射，以匹配你的需求 (prev_count_A, prev_count_A_neg)
        rename_map[pos_col_name] = f'prev_count_{cat}'
        rename_map[neg_col_name] = f'prev_count_{cat}_neg'
    # 步骤 3: 按天聚合 (现在聚合的是 _pos 和 _neg 列)
    df_daily_agg = df_dummies.groupby(['cust_no', 'event_date'])[outcome_cols].sum().reset_index()
    df_daily_agg = df_daily_agg.sort_values(by=['cust_no', 'event_date'])
    # 步骤 4: 计算历史累计 (对 _pos 和 _neg 列操作)
    daily_cumsum = df_daily_agg.groupby('cust_no')[outcome_cols].cumsum()
    # 步骤 5: 获取“截至昨日”的计数
    prev_counts = daily_cumsum.groupby(df_daily_agg['cust_no']).shift(1).fillna(0)
    # 步骤 6: 重命名 (使用我们之前创建的 rename_map)
    prev_counts = prev_counts.rename(columns=rename_map)
    # 步骤 7: 合并与清理 (这部分逻辑不变)
    df_features_by_day = pd.concat([df_daily_agg[['cust_no', 'event_date']], prev_counts], axis=1)
    df_final = pd.merge(df_processed, df_features_by_day, on=['cust_no', 'event_date'], how='left')
    df_final = df_final.sort_values(by='_original_sort_order')
    df_final = df_final.drop(columns='_original_sort_order')
    return df_final

# ------------------ 客户表清洗 ------------------
def clean_cust_df(path):
    df = pd.read_csv(path,dtype=object)
    df = df.replace(r'^\s*$',pd.NA,regex=True)

    # 性别处理
    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper().map({'M':1,'F':0}).fillna(0.5)

    # 年龄处理
    if 'birth_ym' in df.columns:
        def parse_age(x):
            try:
                s = str(x)
                if len(x) >= 4:
                    y = x.split('/')[0]
                    return 2024 - int(y)
            except:
                return 35
            return 35
        df['age'] = df['birth_ym'].apply(parse_age)
    else:
        df['age'] = 35

    # 教育程度 / 婚姻情况
    if 'edu_bg' in df.columns:
        df['edu_bg'] = pd.to_numeric(df['edu_bg'], errors='coerce')
        df['edu_bg'] = df['edu_bg'].fillna(df['edu_bg'].mean())
    if 'marriage_situ_cd' in df.columns:
        df['marriage_situ_cd'] = pd.to_numeric(df['marriage_situ_cd'], errors='coerce')
        df['marriage_situ_cd'] = df['marriage_situ_cd'].fillna(df['marriage_situ_cd'].mean())

    # 保留必要列
    keep_cols = ['cust_no', 'gender', 'age', 'edu_bg', 'marriage_situ_cd']
    df = df[keep_cols]
    print(f"[CLEAN] cust_df cleaned: {df.shape}, columns={df.columns.tolist()}")
    return df

# ------------------ 事件表清洗 ------------------
def clean_event_df(path):
    df = pd.read_csv(path,dtype=object)

    # 标准化ID
    df['prod_id'] = df['prod_id'].astype(str).apply(normalize_id_str)
    df['cust_no'] = df['cust_no'].astype(str).apply(normalize_id_str)
    df['event_type'] = df['event_type'].astype(str).str.strip()
    df['event_level'] = df['event_level'].astype(str).str.strip()
    # event_type独热编码
    df['A'] = df['event_type'].apply(lambda x: 1 if str(x).upper() == 'A' else 0)
    df['B'] = df['event_type'].apply(lambda x: 1 if str(x).upper() == 'B' else 0)
    df['D'] = df['event_type'].apply(lambda x: 1 if x == 'D' else 0)
    df['is_success'] = df['event_type'].apply(lambda x: 1 if x in ['A', 'B'] else 0)
    # event_level独热编码
    df['event_level_A'] = df['event_level'].apply(lambda x: 1 if str(x).upper() == 'A' else 0)
    df['event_level_B'] = df['event_level'].apply(lambda x: 1 if str(x).upper() == 'B' else 0)
    df['event_level_C'] = df['event_level'].apply(lambda x: 1 if str(x).upper() == 'C' else 0)
    # 保留 event_level 和 event_date 原值
    if 'event_level' in df.columns:
        df['event_level'] = df['event_level'].fillna('unknown')
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    for c in df.columns:
        if c in ['prod_id','cust_no','event_type','event_level','A','B','D','is_success','event_dt']: continue
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['event_date'] = pd.to_datetime(df["event_date"], errors='coerce')

    # 取出prod_cat
    df['prod_cat'] = df['prod_id'].apply(lambda x: str(x).strip()[0])
    df['prod_cat'] = df['prod_cat'].str.upper()
    # 加入前述行为
    df = compute_previous_event_counts_with_success(df)
    print(f"[CLEAN] event_df cleaned: {df.shape}, columns={df.columns.tolist()}")
    return df

# ------------------ 产品表清洗 ------------------
def clean_product_xlsx(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    raws = []
    for sname,df in sheets.items():
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
