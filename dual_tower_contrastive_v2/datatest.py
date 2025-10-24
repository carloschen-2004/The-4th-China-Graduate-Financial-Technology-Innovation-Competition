import pandas as pd

import torch
from models.user_tower import UserTower
from models.product_tower import ProductTower

cust = pd.read_csv("data/cleaned/cleaned_cust_dataset.csv")
prod = pd.read_excel("data/cleaned/cleaned_productLabels_multiSpreadsheets.xlsx")

print("cust numeric describe:")
print(cust.select_dtypes(include='number').describe())
print("prod numeric describe:")
print(prod.select_dtypes(include='number').describe())


chk = torch.load("outputs/model_D.pth", map_location='cpu')
u_t = UserTower(len(chk['user_cols']), embed_dim=64)
u_t.load_state_dict(chk['user_state'])
print("user_tower first-layer weights std:", float(u_t.net[0].weight.std()))
