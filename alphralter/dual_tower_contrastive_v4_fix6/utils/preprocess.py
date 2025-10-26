import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_users_cust_all(cust_df):
    df = cust_df.copy()

    # 确保cust_no存在
    if 'cust_no' not in df.columns:
        raise ValueError("cust_no not found in cust_df")

    # ========== 性别处理 ==========
    # 支持 gender/sex/性别 等列名
    gender_alias = [c for c in df.columns if any(x in str(c).lower() for x in ['gender', 'sex', '性别'])]
    if gender_alias:
        df.rename(columns={gender_alias[0]: 'gender'}, inplace=True)
        df['gender'] = df['gender'].astype(str).str.upper().map({'M': 1, 'F': 0}).fillna(0.5)
    else:
        df['gender'] = 0.5  # 未提供则默认中性

    # ========== 年龄计算 ==========
    # 支持 'birth_ym', 'birth', 'birthday' 等列
    birth_alias = [c for c in df.columns if any(x in str(c).lower() for x in ['birth', 'birthday', '出生'])]
    if birth_alias:
        def parse_birth(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            # 处理如 'Mar-42' 'Aug-86' '1990-05' 等
            try:
                if '-' in s and len(s.split('-')[1]) <= 2:
                    # 解析月-年形式
                    dt = pd.to_datetime(s, format='%b-%y', errors='coerce')
                    if pd.isna(dt):
                        dt = pd.to_datetime(s, errors='coerce')
                else:
                    dt = pd.to_datetime(s, errors='coerce')
                if pd.notna(dt):
                    return dt.year
            except Exception:
                pass
            # 若无法解析，则尝试取数字部分
            import re
            digits = re.findall(r'\\d{2,4}', s)
            if digits:
                y = int(digits[-1])
                if y < 100:
                    y += 1900
                return y
            return np.nan

        df['birth_year'] = df[birth_alias[0]].apply(parse_birth)
        current_year = datetime.now().year
        df['age'] = (current_year - df['birth_year']).fillna(35).clip(lower=18, upper=100)
    else:
        df['age'] = 35  # 默认年龄

    # ========== 其他基础列处理 ==========
    for c in df.columns:
        if c == 'cust_no':
            continue
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

    # 确保 edu_bg / marriage_situ_cd 存在
    for col in ['edu_bg', 'marriage_situ_cd']:
        if col not in df.columns:
            df[col] = 0.0

    # ========== 输出 ==========
    feat_cols = [c for c in df.columns if c != 'cust_no']
    df = df[['cust_no'] + feat_cols]
    return df, feat_cols
