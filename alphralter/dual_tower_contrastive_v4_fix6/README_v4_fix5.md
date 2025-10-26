Dual-Tower Contrastive Recommendation v4_fix5
--------------------------------------------------

fix5：

强制使用训练模型里保存的 prod_cols 与 user_cols（若某些列在 cust_df/prod_df 缺失会自动补 0）。

优先使用事件中活跃用户（若事件中有真实购买用户，会把这些用户的 embedding 也计算进来并优先候选）。

当用户 embedding 或相似度退化（接近常数）时，回退到事件聚合（按购买次数）作为 topK，避免每个产品都一样。

统计画像（性别 / 年龄 / 教育 / 婚姻）时做了安全取值（列不存在或均为 0 会有告警）。

输出每个产品还附带 fallback_reason 字段（如果发生回退），便于排查。

1. 三个数据文件放到data/: cust_dataset.csv, event_dataset.csv, productLabels_multiSpreadsheets.xlsx
2. 虚拟环境+依赖： requirements.txt
3. 训练:
   python train_per_category.py --data_dir data --save_dir outputs --epochs 2 --batch_size 64 --lr 3e-4
4. 推荐:
   python recommend_user_profiles.py --data_dir data --model_dir outputs --out_csv outputs/recommended_user_profiles.csv --k 200

输出结果在outputs里

pth文件是模型

xlsx是推荐结果
