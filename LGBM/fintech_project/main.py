#!/usr/bin/env python3
# main.py — 主入口（将项目根加入 sys.path，避免包导入问题）

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from utils.logger import get_logger
from pipeline.multi_step import train_and_evaluate

logger = get_logger("main", logfile="main.log")

def main():
    parser = argparse.ArgumentParser(description="FinTech two-stage recommendation pipeline")
    parser.add_argument("--n_customers", type=int, default=None, help="Use first N customers (for speed)")
    parser.add_argument("--n_products", type=int, default=None, help="Use first M products (for speed)")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate for cust/events if using real files")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP and importance plots for step2")
    args = parser.parse_args()

    logger.info(f"Starting pipeline with n_customers={args.n_customers}, n_products={args.n_products}, sample_rate={args.sample_rate}, explain={args.explain}")
    res = train_and_evaluate(
        simulate=False,
        n_customers=args.n_customers,
        n_products=args.n_products,
        explain=args.explain,
        sample_rate=args.sample_rate
    )

    logger.info("Pipeline finished. Metrics:")
    for k, v in res["metrics"].items():
        logger.info(f"{k}: {v}")
    # show a small slice of recommendation merged file if present in return
    recs = res.get("recommendations_merged")
    if recs is not None:
        print("\n🧾 推荐（每客户一行）示例：")
        print(recs.head(10).to_string(index=False))
    print("\nDone. See logs/ and outputs/ for artifacts.")

if __name__ == "__main__":
    main()
