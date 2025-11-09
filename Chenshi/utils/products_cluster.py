import pandas as pd

# 注意修改路径
input_file = "/Users/keep-rational/Desktop/研创赛-金科-多步推荐/productLabels_multiSpreadsheets.xlsx"
output_file = "/Users/keep-rational/Desktop/研创赛-金科-多步推荐/Chenshi/data/first_cluster.xlsx"
product_column = "prod_id"

# 读取xlsx文件
all_sheets = pd.read_excel(input_file, sheet_name=None)
# 2. 创建一个空字典，用于存储每个sheet处理后的结果
grouped_results = {}
# 3. 逐个sheet处理
for sheet_name, df in all_sheets.items():
    print(f"正在处理 sheet:{sheet_name} ...")
    if product_column not in df.columns:
        print(f"⚠️ 跳过 {sheet_name}：未找到列 '{product_column}'")
        continue
    feature_cols = [col for col in df.columns if col != product_column]
    grouped = (
        df.groupby(feature_cols, dropna=False)[product_column]
        .apply(list)
        .reset_index()
        .rename(columns={product_column: "prod_id"})
    )
    cols = ["prod_id"] + [c for c in grouped.columns if c != "prod_id"]
    grouped = grouped[cols]
    grouped = grouped.sort_values(by="prod_id", key=lambda x: x.str[0])
    grouped["first_cluster_id"] = [str(sheet_name)+ str(idx) for idx, _ in enumerate(grouped["prod_id"])]
    grouped_results[sheet_name] = grouped

# 4. 将所有结果写入新的Excel文件
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, grouped_df in grouped_results.items():
        grouped_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"✅ 全部处理完成！结果已保存为：{output_file}")