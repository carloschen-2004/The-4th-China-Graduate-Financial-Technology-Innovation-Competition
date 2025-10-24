import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_users(cust_df, reference_year=None):
    df = cust_df.copy()
    if 'birth_ym' in df.columns:
        def extract_year(x):
            try:
                s = str(x)
                if len(s) >= 4:
                    return int(s[:4])
            except:
                return None
            return None
        df['birth_year'] = df['birth_ym'].apply(extract_year).fillna(0).astype(int)
    else:
        df['birth_year'] = 0
    if reference_year is None:
        reference_year = datetime.now().year
    df['age'] = reference_year - df['birth_year']
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'M':1, 'F':0, '男':1, '女':0}).fillna(df['gender'])
    else:
        df['gender'] = 0
    if 'edu_bg' in df.columns:
        df['edu_bg'] = pd.factorize(df['edu_bg'].astype(str))[0]
    else:
        df['edu_bg'] = 0
    if 'marriage_situ_cd' in df.columns:
        df['marriage_situ_cd'] = pd.factorize(df['marriage_situ_cd'].astype(str))[0]
    else:
        df['marriage_situ_cd'] = 0
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median()).astype(float)
    df['gender'] = pd.to_numeric(df['gender'], errors='coerce').fillna(0).astype(float)
    df['edu_bg'] = pd.to_numeric(df['edu_bg'], errors='coerce').fillna(0).astype(float)
    df['marriage_situ_cd'] = pd.to_numeric(df['marriage_situ_cd'], errors='coerce').fillna(0).astype(float)
    return df
