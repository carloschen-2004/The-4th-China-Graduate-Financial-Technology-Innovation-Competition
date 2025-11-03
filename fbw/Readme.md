# 双塔模型 + Transformer 

### 基本运行

```
# 使用默认参数运行完整流水线
python pipeline.py

# 指定数据目录和工作目录
python pipeline.py --data_dir ./my_data --work_dir ./my_outputs
```

### 数据路径参数

- `--data_dir`: 输入数据目录路径（默认：`data`）
- `--work_dir`: 共享工作目录路径（默认：`outputs`）

### 双塔模型训练参数

- `--epochs`: 训练轮数（默认：`2`）
- `--batch_size`: 批次大小（默认：`64`）
- `--lr`: 学习率（默认：`0.0003`）
- `--k`: 每个商品推荐用户数量（默认：`200`）

### Transformer模型参数

- `--transformer_epochs`: Transformer训练轮数（默认：`5`）

### 输入文件（放在`data_dir`目录下）

```
data_dir/
├── cust_dataset.csv              # 客户数据集
├── event_dataset.csv             # 事件数据集
└── productLabels_multiSpreadsheets.xlsx  # 商品标签数据
```