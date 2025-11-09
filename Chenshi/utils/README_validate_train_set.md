# validate_train_set 使用说明

## 功能概述

`validate_train_set` 函数用于增强训练集，通过双塔模型的embeddings为每个用户生成top-K推荐产品，并在训练集中补充负样本（D类型事件）。

## 函数签名

```python
def validate_train_set(
    cust_df,           # 客户数据框
    prod_df,           # 产品数据框
    events_df,         # 事件数据框
    user_emb,          # 用户embedding矩阵 (n_users, emb_dim)
    prod_emb,          # 产品embedding矩阵 (n_products, emb_dim)
    user_ids,          # 用户ID数组，与user_emb的行对应
    prod_ids,          # 产品ID数组，与prod_emb的行对应
    top_k=200,         # 每个用户取top-K个推荐产品，默认200
    out_dir='outputs', # 输出目录，默认'outputs'
    out_name='modified_train_events.csv'  # 输出文件名
)
```

## 使用方法

### 1. 基本使用（在Notebook中）

```python
import pandas as pd
import numpy as np
from utils.validate_train_set import validate_train_set

# 加载数据
cust_df = pd.read_csv('../cust_dataset.csv')
events_df = pd.read_csv('../event_dataset.csv')
prod_df = pd.read_excel('../productLabels_multiSpreadsheets.xlsx', sheet_name=None)

# 准备embeddings（需要从双塔模型获取）
# user_emb: (n_users, emb_dim) numpy数组
# prod_emb: (n_products, emb_dim) numpy数组
# user_ids: (n_users,) 用户ID数组
# prod_ids: (n_products,) 产品ID数组

# 调用函数
modified_events = validate_train_set(
    cust_df=cust_df,
    prod_df=prod_df,
    events_df=events_df,
    user_emb=user_emb,
    prod_emb=prod_emb,
    user_ids=user_ids,
    prod_ids=prod_ids,
    top_k=200,
    out_dir='outputs',
    out_name='modified_train_events.csv'
)
```

### 2. 从双塔模型获取embeddings

如果你使用的是双塔模型（如 `alphralter/dual_tower_contrastive_v5`），可以参考以下代码：

```python
import torch
from models.user_tower import UserTower
from models.product_tower import ProductTower
from sklearn.preprocessing import normalize

# 加载模型
model_path = 'path/to/your/model.pth'
model_state = torch.load(model_path, map_location='cpu')

user_tower = UserTower(...)
product_tower = ProductTower(...)
user_tower.load_state_dict(model_state['user_tower'])
product_tower.load_state_dict(model_state['product_tower'])

# 准备特征
user_features = prepare_user_features(cust_df)  # 需要根据实际情况实现
product_features = prepare_product_features(prod_df)  # 需要根据实际情况实现

# 生成embeddings
with torch.no_grad():
    user_emb = user_tower(torch.tensor(user_features, dtype=torch.float32)).numpy()
    prod_emb = product_tower(torch.tensor(product_features, dtype=torch.float32)).numpy()

# 归一化embeddings（推荐）
user_emb = normalize(user_emb, axis=1)
prod_emb = normalize(prod_emb, axis=1)

# 准备ID数组
user_ids = cust_df['cust_no'].values.astype(str)
prod_ids = prod_df['prod_id'].values.astype(str)
```

### 3. 与Transformer模型结合使用

如果你使用的是Transformer模型而不是双塔模型，你需要：

1. 从Transformer模型中提取用户和产品的表示
2. 或者使用其他方法生成embeddings（如通过中间层输出）
3. 然后将这些embeddings传递给 `validate_train_set` 函数

## 输出说明

函数会：
1. 为每个用户计算top-K推荐产品
2. 检查这些推荐产品在训练集中是否已有成功事件（A或B类型）
3. 如果没有，则添加D类型事件作为负样本
4. 将增强后的训练集保存到指定文件
5. 返回修改后的数据框

## 注意事项

1. **embeddings维度**: `user_emb` 和 `prod_emb` 的embedding维度必须相同
2. **ID对应关系**: `user_ids` 必须与 `user_emb` 的行一一对应，`prod_ids` 必须与 `prod_emb` 的行一一对应
3. **数据格式**: `events_df` 必须包含 `cust_no`、`prod_id` 和 `event_type` 列
4. **输出目录**: 函数会自动创建输出目录（如果不存在）

## 示例输出

```
所有唯一的产品类别 (Unique prod_cat): ['A', 'C', 'D', 'N', 'P']
[DONE] saved modified training events to outputs/modified_train_events.csv, rows=350000
```

## 相关文件

- `validate_train_set.py`: 主函数文件
- `validate_train_set_example.py`: 使用示例代码
- `data_cleaning.py`: 数据清洗函数（用于准备输入数据）



