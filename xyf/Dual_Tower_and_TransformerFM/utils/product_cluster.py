import pandas as pd
import os

# work_dir: Dual_Tower_and_TransformerFM
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(os.path.dirname(current_script_path))

def prod_cluster():
    input_file = os.path.join(work_dir, "raw_data/productLabels_multiSpreadsheets.xlsx")
    output_file = os.path.join(work_dir, "cluster_products/first_cluster.xlsx")
    product_column = "prod_id"
    # 读取xlsx文件
    all_sheets = pd.read_excel(input_file, sheet_name=None)
    # 2. 创建一个空字典，用于存储每个sheet处理后的结果
    grouped_results = {}
    # 3. 逐个sheet处理
    for sheet_name, df in all_sheets.items():
        print(f"正在处理 sheet：{sheet_name} ...")
        # 检查产品名称列是否存在
        if product_column not in df.columns:
            print(f"⚠️ 跳过 {sheet_name}：未找到列 '{product_column}'")
            continue
        # 获取特征列（除产品列外）
        feature_cols = [col for col in df.columns if col != product_column]
        # 按特征列分组，收集产品列表
        grouped = (
            df.groupby(feature_cols, dropna=False)[product_column]
            .apply(list)
            .reset_index()
            #.rename(columns={product_column: "prod_id_list"})
        )
        # 调整列顺序 & 按第一个产品排序
        cols = ["prod_id"] + [c for c in grouped.columns if c != "prod_id"]
        grouped = grouped[cols]
        grouped = grouped.sort_values(by="prod_id", key=lambda x: x.str[0])

        # 存入结果字典
        grouped_results[sheet_name] = grouped

    # 4. 将所有结果写入新的Excel文件
    os.makedirs(os.path.join(work_dir, "cluster_products"), exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, grouped_df in grouped_results.items():
            grouped_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"✅ 全部处理完成！结果已保存为：{output_file}")

if __name__ == "__main__":
    prod_cluster()