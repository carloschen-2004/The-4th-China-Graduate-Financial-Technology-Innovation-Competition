Dual-Tower Contrastive Recommendation v4_fix3
--------------------------------------------------

fix3：

- 客户特征：使用 cust_dataset.csv 中的所有列，除 ID 列 (cust_no) 外。缺失值用 0 填充
- 事件成功：event_type 值为 'A' 或 'B' 被认为是成功事件（根据 数据集说明.docx）
- 数据处理：自动检测 ID 列，强制转换类型，删除零方差特征，对数值特征进行标准化
- 对产品特征添加少量抖动，以避免完全恒定的输入导致 loss=0。- 保存 prod_cols/prod_dim/user_cols 与模型
- 日志记录和检查，以避免 “[TRAIN] No successful events for...” 和 “loss=0” 问题

1. 三个数据文件放到data/: cust_dataset.csv, event_dataset.csv, productLabels_multiSpreadsheets.xlsx
2. 虚拟环境+依赖： requirements.txt
3. 训练:
   python train_per_category.py --data_dir data --save_dir outputs --epochs 2 --batch_size 64 --lr 3e-4
4. 推荐:
   python recommend_user_profiles.py --data_dir data --model_dir outputs --out_csv outputs/recommended_user_profiles.csv --k 200

