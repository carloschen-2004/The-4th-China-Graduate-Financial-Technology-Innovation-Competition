# data/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from utils.logger import get_logger

logger = get_logger("feature_engineering")

def bin_continuous(df, cols, n_bins=5):
    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    sub = df[cols].fillna(0)
    try:
        arr = kb.fit_transform(sub)
        for i, c in enumerate(cols):
            df[f"{c}_bin"] = arr[:,i].astype(int)
    except Exception as e:
        logger.warning(f"KBins failed: {e}")
        for c in cols:
            df[f"{c}_bin"] = pd.qcut(df[c].rank(method='first'), q=n_bins, labels=False, duplicates='drop').astype(int)
    return df

def build_marketing_tag(events_df):
    # last two events per user -> combine event_id,event_type,event_level
    df = events_df.sort_values(['cust_no','event_date'])
    last2 = df.groupby('cust_no').tail(2)
    pivot = last2.assign(rank=last2.groupby('cust_no')['event_date'].rank(method='first')).pivot(
        index='cust_no', columns='rank', values='event_id'
    )
    pivot = pivot.fillna('NA').astype(str)
    pivot['marketing_tag'] = pivot.apply(lambda row: "__".join(row.values.tolist()), axis=1)
    pivot = pivot.reset_index()[['cust_no','marketing_tag']]
    return pivot

def build_features(df_merged, events_df):
    df = df_merged.copy()
    # continuous binning example
    cont_cols = []
    if 'total_events' in df.columns:
        cont_cols.append('total_events')
    if 'success_events' in df.columns:
        cont_cols.append('success_events')
    if cont_cols:
        df = bin_continuous(df, cont_cols, n_bins=5)
    # marketing tag
    mtag = build_marketing_tag(events_df)
    df = df.merge(mtag, on='cust_no', how='left')
    df['marketing_tag'] = df['marketing_tag'].fillna('NA')
    # encode some categorical to numeric simple way
    for c in ['loc_cd','gender','edu_bg','marriage_situ_cd','prod_category']:
        if c in df.columns:
            df[c] = df[c].fillna('NA').astype(str)
            df[c+'_enc'] = df[c].astype('category').cat.codes
    # prepare feature list
    features = []
    for f in ['prod_id', 'new_flag', 'prod_category', 'total_events', 'success_events', 'marketing_tag']:
        if f in df.columns:
            if f == 'marketing_tag':
                df['marketing_tag_freq'] = df.groupby('marketing_tag')['marketing_tag'].transform('count')
                features.append('marketing_tag_freq')
            elif f == 'prod_category':
                features.append('prod_category_enc' if 'prod_category_enc' in df.columns else None)
            elif f == 'prod_id':
                df['prod_id_enc'] = df['prod_id'].astype('category').cat.codes
                features.append('prod_id_enc')
            else:
                features.append(f)
    # add some encoded static cols
    for enc in ['loc_cd_enc','gender_enc','edu_bg_enc','marriage_situ_cd_enc']:
        if enc in df.columns:
            features.append(enc)
    features = [f for f in features if f is not None]
    logger.info(f"Feature list: {features}")
    return df, features
