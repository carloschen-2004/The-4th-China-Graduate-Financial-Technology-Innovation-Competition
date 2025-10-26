Dual-Tower Contrastive Recommendation v4_fix4
--------------------------------------------------

fix4：

- 年龄读取年月格式丢失修复，用birth和init计算
- 性别字符串转int

1. 三个数据文件放到data/: cust_dataset.csv, event_dataset.csv, productLabels_multiSpreadsheets.xlsx
2. 虚拟环境+依赖： requirements.txt
3. 训练:
   python train_per_category.py --data_dir data --save_dir outputs --epochs 2 --batch_size 64 --lr 3e-4
4. 推荐:
   python recommend_user_profiles.py --data_dir data --model_dir outputs --out_csv outputs/recommended_user_profiles.csv --k 200

问题：某些 product 的相似度矩阵退化（常见原因：用户或产品向量恒常
A类的193个产品的训练结果完全相同