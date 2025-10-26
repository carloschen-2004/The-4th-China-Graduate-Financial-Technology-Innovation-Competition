Dual-Tower Contrastive Recommendation (stable v2)双塔v2

- 缺失的项自动补零（婚姻教育那些）
- 按类别的模型（D/C/A/N/P）
- 数值稳定性修复（eps、限制值、梯度裁剪）
- 在检查点中保存 prod_cols 和 prod_dim 以实现正确加载

记得往data/下面放那三个数据文件！！！

用法（epochs那些的默认值在train_per_category文件里）:
  pip install -r requirements.txt
  python3 train_per_category.py --data_dir data --save_dir outputs --epochs 5 --batch_size 256 --lr 1e-4 --temperature 0.7
  python recommend_user_profiles.py --data_dir data --model_dir outputs --out_csv outputs/recommended_user_profiles.csv --k 200

目前loss是零，各位看看能不能修好它……（调参似乎也没用，我猜测是婚姻和教育补零太多污染了有效样本）

![image-20251024193445343](C:\Users\yuan_ci\AppData\Roaming\Typora\typora-user-images\image-20251024193445343.png)

我尽力了，v1版的训练loss甚至是nan……（挠头）

![image-20251024193524523](C:\Users\yuan_ci\AppData\Roaming\Typora\typora-user-images\image-20251024193524523.png)
