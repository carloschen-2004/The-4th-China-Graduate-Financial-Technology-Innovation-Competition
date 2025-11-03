import argparse
import os
import subprocess
import sys
import shutil


def run_cmd(cmd: list, cwd: str | None = None, env: dict | None = None) -> None:
    # 执行外部命令（可指定工作目录与环境变量）
    print(f"[运行] {' '.join(cmd)}")
    result_env = os.environ.copy()
    if env is not None:
        result_env.update(env)
    result = subprocess.run(cmd, cwd=cwd, env=result_env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="先运行双塔，再运行Chenshi的Transformer（notebook）")
    parser.add_argument("--data_dir", default="data", help="双塔输入数据目录")
    parser.add_argument("--work_dir", default="outputs", help="双塔输出与Transformer输入的共享目录")
    parser.add_argument("--epochs", type=int, default=2, help="双塔训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="双塔批大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="双塔学习率")
    parser.add_argument("--k", type=int, default=200, help="每个产品选Top-K客户")
    parser.add_argument("--transformer_epochs", type=int, default=5, help="Transformer训练轮数（如notebook使用该参数）")
    args = parser.parse_args()

    # 统一绝对路径
    root_dir = os.getcwd()
    abs_data = os.path.abspath(args.data_dir)
    abs_work = os.path.abspath(args.work_dir)
    os.makedirs(abs_work, exist_ok=True)

    # 1) 训练双塔（按品类）
    run_cmd([
        sys.executable,
        "train_per_category.py",
        "--data_dir", abs_data,
        "--save_dir", abs_work,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
    ], cwd=os.path.join(os.getcwd(), "dual_tower_contrastive_v5"))

    # 2) 画像导出（每个产品的Top-K客户），写入共享目录
    out_csv = os.path.join(abs_work, "recommended_user_profiles.csv")
    run_cmd([
        sys.executable,
        "recommend_user_profiles.py",
        "--data_dir", abs_data,
        "--model_dir", abs_work,
        "--out_csv", out_csv,
        "--k", str(args.k),
    ], cwd=os.path.join(os.getcwd(), "dual_tower_contrastive_v5"))

    # 3) 直接Transformer notebook
    chenshi_dir = os.path.join(root_dir, "Chenshi")
    nb_path = os.path.join(chenshi_dir, "Transformer_baseline.ipynb")
    # 为兼容 notebook 内的相对路径（如 '../cust_dataset.csv'），在工程根目录准备三份数据副本
    src_files = [
        (os.path.join(abs_data, 'cust_dataset.csv'), os.path.join(root_dir, 'cust_dataset.csv')),
        (os.path.join(abs_data, 'event_dataset.csv'), os.path.join(root_dir, 'event_dataset.csv')),
        (os.path.join(abs_data, 'productLabels_multiSpreadsheets.xlsx'), os.path.join(root_dir, 'productLabels_multiSpreadsheets.xlsx')),
    ]
    for src, dst in src_files:
        if os.path.exists(src):
            # 若不存在或源文件更新，则复制
            if (not os.path.exists(dst)) or (os.path.getmtime(src) > os.path.getmtime(dst)):
                shutil.copy2(src, dst)

    # 通过环境变量传递双塔输出路径（notebook如需可读取）
    nb_env = {"PROFILES_CSV": out_csv, "WORK_DIR": abs_work, "EPOCHS": str(args.transformer_epochs)}
    try:
        # 优先使用 papermill（输出到绝对路径）
        run_cmd([sys.executable, "-m", "papermill", nb_path, os.path.join(abs_work, "Transformer_baseline.out.ipynb")], cwd=chenshi_dir, env=nb_env)
    except SystemExit:
        # 退回使用 jupyter nbconvert（输出到绝对路径）
        run_cmd([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", nb_path, "--output", os.path.join(abs_work, "Transformer_baseline.out.ipynb")], cwd=chenshi_dir, env=nb_env)


if __name__ == "__main__":
    main()


