# data/data_loader.py
"""
Data loader supporting:
- cust_dataset.csv
- event_dataset.csv
- productLabels_multiSpreadsheets.xlsx (sheets D,C,A,N,P with differing columns)

If the real files under data/raw/ are missing, this module falls back to
generating simulated data with the same field layouts so the pipeline can run.
"""

import os
import pandas as pd
import numpy as np
from config import DATA_RAW, RANDOM_SEED, SAMPLE_RATE
from utils.logger import get_logger

logger = get_logger("data_loader", logfile="data_loader.log")
np.random.seed(RANDOM_SEED)

# expected product sheet columns map
_SHEET_MAP = {
    "D": ["prod_id", "interval_level", "deposit_type1", "deposit_type2"],
    "C": ["prod_id", "credit_level", "credit_amt_cd", "credit_type1", "credit_type2", "new_flag"],
    "A": ["prod_id", "frtn_type", "fr_period_type", "fr_prod_attr", "fr_prod_type", "fr_risk_level"],
    "N": ["prod_id", "channel_type", "channel_type2"],
    "P": ["prod_id", "pay_type1", "pay_type2"],
}


# ----------------------------
# Read customers
# ----------------------------
def read_cust(path=None, sample_rate=SAMPLE_RATE):
    path = path or os.path.join(DATA_RAW, "cust_dataset.csv")
    if os.path.exists(path):
        logger.info(f"Loading customer file: {path}")
        df = pd.read_csv(path, dtype=str)
        if "cust_no" not in df.columns:
            raise ValueError("cust_dataset must contain cust_no")
        df["cust_no"] = df["cust_no"].astype(str)
        if "init_dt" in df.columns:
            df["init_dt"] = pd.to_datetime(df["init_dt"], errors="coerce")
        for c in ["loc_cd", "gender", "edu_bg", "marriage_situ_cd"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
    else:
        logger.warning(f"Customer file not found at {path}. Generating synthetic customer sample.")
        n = 100000 if sample_rate >= 1.0 else int(100000 * sample_rate)
        df = pd.DataFrame({
            "cust_no": [f"C{idx:07d}" for idx in range(1, n+1)],
            "birth_ym": np.random.choice(range(1960,2005), n),
            "loc_cd": np.random.choice(["110","310","320","440"], n),
            "gender": np.random.choice(["M","F"], n),
            "init_dt": pd.to_datetime("2015-01-01") + pd.to_timedelta(np.random.randint(0,3000,n), unit="D"),
            "edu_bg": np.random.choice(["E1","E2","E3","E4"], n),
            "marriage_situ_cd": np.random.choice(["M1","M2"], n)
        })
    if sample_rate < 1.0 and os.path.exists(path):
        df = df.sample(frac=sample_rate, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.info(f"Sampled customers at rate {sample_rate}: {df.shape}")
    logger.info(f"Loaded customers: {df.shape}")
    return df


# ----------------------------
# Read events
# ----------------------------
def read_events(path=None, sample_rate=SAMPLE_RATE):
    path = path or os.path.join(DATA_RAW, "event_dataset.csv")
    if os.path.exists(path):
        logger.info(f"Loading events file: {path}")
        df = pd.read_csv(path, dtype=str)
        if "cust_no" not in df.columns or "event_type" not in df.columns:
            raise ValueError("event_dataset must contain cust_no and event_type")
        df["cust_no"] = df["cust_no"].astype(str)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
        # label success
        df["is_success"] = df["event_type"].isin(["A","B"]).astype(int)
        if sample_rate < 1.0:
            df = df.sample(frac=sample_rate, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        logger.warning(f"Event file not found at {path}. Generating synthetic event sample.")
        # default synthetic sizes
        n_events = int(300000 * sample_rate) if sample_rate >= 1.0 else int(300000 * sample_rate)
        cust_ids = [f"C{idx:07d}" for idx in range(1, 100001)]
        prod_ids = [f"P{idx:05d}" for idx in range(1, 500)]
        df = pd.DataFrame({
            "cust_no": np.random.choice(cust_ids, n_events),
            "prod_id": np.random.choice(prod_ids, n_events),
            "event_id": np.random.choice(["E0001","E0002","E0004","E0007","E0015"], n_events),
            "event_type": np.random.choice(["A","B","D"], n_events, p=[0.3,0.3,0.4]),
            "event_level": np.random.choice(["L1","L2","L3"], n_events),
            "event_date": pd.date_range("2024-01-01", periods=n_events, freq="T"),
            "event_term": np.random.choice([3,6,12,24], n_events),
            "event_rate": np.random.choice([1.5,2.0,2.5,3.0], n_events),
            "event_amt": np.random.choice([1000,5000,10000,50000], n_events)
        })
        df["is_success"] = df["event_type"].isin(["A","B"]).astype(int)
    logger.info(f"Loaded events: {df.shape}, successes: {df['is_success'].sum() if 'is_success' in df.columns else 'N/A'}")
    return df


# ----------------------------
# Read products (multi-sheet)
# ----------------------------
def read_products(path=None):
    path = path or os.path.join(DATA_RAW, "productLabels_multiSpreadsheets.xlsx")
    frames = []
    if os.path.exists(path):
        logger.info(f"Loading product Excel: {path}")
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            if sheet not in _SHEET_MAP:
                # still load but mark
                try:
                    df = xls.parse(sheet_name=sheet, dtype=str)
                    df["prod_category"] = sheet
                    frames.append(df)
                    logger.warning(f"Loaded unexpected sheet {sheet} (kept as-is).")
                except Exception as e:
                    logger.warning(f"Failed to parse sheet {sheet}: {e}")
                continue
            df = xls.parse(sheet_name=sheet, dtype=str)
            df["prod_category"] = sheet
            # only keep expected columns if exist
            expected = _SHEET_MAP[sheet]
            keep = [c for c in expected if c in df.columns]
            # ensure prod_id present
            if "prod_id" not in df.columns:
                logger.warning(f"Sheet {sheet} missing prod_id — skipping sheet")
                continue
            df = df[keep + ["prod_category"]]
            frames.append(df)
            logger.info(f"Loaded product sheet {sheet}: {df.shape}")
    else:
        logger.warning(f"Product Excel not found at {path}. Generating synthetic product pages.")
        # generate synthetic pages consistent with _SHEET_MAP
        for sheet, cols in _SHEET_MAP.items():
            n = 200
            base = {"prod_id": [f"{sheet}{i:04d}" for i in range(1, n+1)]}
            for c in cols:
                if c == "prod_id":
                    continue
                if c == "new_flag":
                    base[c] = np.random.choice([0,1], n, p=[0.8,0.2])
                else:
                    # simple categorical choices
                    base[c] = np.random.choice([f"{c}_v1", f"{c}_v2", f"{c}_v3"], n)
            df = pd.DataFrame(base)
            df["prod_category"] = sheet
            frames.append(df)
    if not frames:
        raise ValueError("No product sheets loaded/constructed.")
    prod_df = pd.concat(frames, ignore_index=True, sort=False)
    prod_df["prod_id"] = prod_df["prod_id"].astype(str)
    if "new_flag" in prod_df.columns:
        prod_df["new_flag"] = pd.to_numeric(prod_df["new_flag"], errors="coerce").fillna(0).astype(int)
    else:
        prod_df["new_flag"] = 0
    logger.info(f"Combined product dataframe: {prod_df.shape}")
    return prod_df


# ----------------------------
# Build positive labels
# ----------------------------
def build_positive_labels(events_df):
    pos = events_df[events_df["is_success"] == 1]
    grouped = pos.groupby(["cust_no","prod_id"], as_index=False).agg(
        first_success_date=("event_date", "min"),
        n_success=("is_success", "sum")
    )
    grouped["is_pos"] = 1
    logger.info(f"Built positive label pairs: {grouped.shape}")
    return grouped[["cust_no","prod_id","is_pos","first_success_date","n_success"]]


# ----------------------------
# Build pairs (neg sampling)
# ----------------------------
def build_user_product_pairs(cust_df, events_df, prod_df, neg_per_pos=50):
    pos = build_positive_labels(events_df)
    all_products = prod_df["prod_id"].unique().tolist()
    cust_ids = cust_df["cust_no"].unique().tolist()
    rows = []
    pos_map = pos.groupby("cust_no")["prod_id"].apply(set).to_dict()
    logger.info("Constructing pairs (this can be memory-heavy)...")
    for idx, cust in enumerate(cust_ids):
        pos_set = pos_map.get(cust, set())
        for p in pos_set:
            rows.append({"cust_no": cust, "prod_id": p, "is_pos": 1})
        neg_candidates = [p for p in all_products if p not in pos_set]
        if len(pos_set) > 0:
            n_neg = max(1, int(len(pos_set) * neg_per_pos))
        else:
            n_neg = min(10, len(all_products))
        n_neg = min(len(neg_candidates), n_neg)
        if n_neg > 0 and len(neg_candidates) > 0:
            sampled = np.random.choice(neg_candidates, size=n_neg, replace=False)
            for p in sampled:
                rows.append({"cust_no": cust, "prod_id": p, "is_pos": 0})
        if idx % 50000 == 0 and idx > 0:
            logger.info(f"Processed {idx} customers, pairs so far: {len(rows)}")
    df_pairs = pd.DataFrame(rows)
    logger.info(f"Constructed pairs: {df_pairs.shape}, positives: {int(df_pairs['is_pos'].sum())}")
    return df_pairs
