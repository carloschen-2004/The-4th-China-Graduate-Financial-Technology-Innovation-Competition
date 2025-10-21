# data/preprocess.py
import pandas as pd
from utils.logger import get_logger

logger = get_logger("preprocess", logfile="preprocess.log")

def merge_for_model(cust_df, events_df, prod_df, pairs_df):
    """
    Merge pairs with customer and product attributes, add aggregated event features.
    """
    df = pairs_df.merge(cust_df, on="cust_no", how="left")
    df = df.merge(prod_df, on="prod_id", how="left")

    # compute simple user-level aggregates from events_df
    user_agg = events_df.groupby("cust_no").agg(
        total_events=("event_id","count"),
        success_events=("is_success","sum"),
        last_event_date=("event_date","max")
    ).reset_index()
    df = df.merge(user_agg, on="cust_no", how="left")
    # fillna
    df["total_events"] = df["total_events"].fillna(0).astype(int)
    df["success_events"] = df["success_events"].fillna(0).astype(int)
    logger.info(f"Merged data for modeling: {df.shape}")
    return df
