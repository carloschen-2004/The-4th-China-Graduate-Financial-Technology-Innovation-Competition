import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.user_tower import UserTower
from models.product_tower import ProductTower
from models.dual_tower import DualTowerContrastive
from utils.data_cleaner import auto_clean_data
from utils.preprocess import preprocess_users
from tqdm import tqdm

class CategoryDataset(Dataset):
    def __init__(self, merged_df, user_cols, prod_cols):
        self.merged = merged_df.reset_index(drop=True)
        self.user_cols = user_cols
        self.prod_cols = prod_cols
        if len(self.prod_cols) == 0:
            self.merged['prod_hash'] = self.merged['prod_id'].astype(str).apply(lambda x: hash(x)%1000)
            self.prod_cols = ['prod_hash']

    def __len__(self):
        return len(self.merged)
    def __getitem__(self, idx):
        row = self.merged.iloc[idx]
        u = row[self.user_cols].values.astype(float)
        p = row[self.prod_cols].values.astype(float) if len(self.prod_cols)>0 else np.array([0.0])
        return u, p

def collate_fn(batch):
    users = torch.tensor([b[0] for b in batch], dtype=torch.float32)
    prods = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return users, prods

def train_for_category(cat, args, cust_df, prod_df, events_df):
    print(f"[TRAIN] Starting category {cat}")
    prod_mask = prod_df['prod_id'].astype(str).str.startswith(cat)
    prod_sub = prod_df[prod_mask].reset_index(drop=True)
    if prod_sub.shape[0] == 0:
        print(f"[TRAIN] No products found for category {cat}"); return
    ev = events_df[events_df['event_type'].isin(['A','B'])].copy()
    ev = ev[ev['prod_id'].astype(str).isin(prod_sub['prod_id'].astype(str))]
    if ev.shape[0] == 0:
        print(f"[TRAIN] No successful events for category {cat}"); return
    merged = ev.merge(cust_df, on='cust_no', how='inner').merge(prod_sub, on='prod_id', how='inner')
    if merged.shape[0] == 0:
        print(f"[TRAIN] Merged dataframe empty for {cat}"); return
    user_cols = ['gender','age','edu_bg','marriage_situ_cd']
    prod_cols = [c for c in prod_sub.columns if c not in ['prod_id','__sheet__','new_flag'] and pd.api.types.is_numeric_dtype(prod_sub[c])]
    ds = CategoryDataset(merged, user_cols, prod_cols)
    if len(ds) < 8:
        print(f"[TRAIN] Too few samples ({len(ds)}) for category {cat}; skipping."); return
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    user_dim = len(user_cols)
    prod_dim = len(prod_cols) if len(prod_cols)>0 else 1
    user_t = UserTower(user_dim, embed_dim=args.embed_dim).to(device)
    prod_t = ProductTower(prod_dim, embed_dim=args.embed_dim).to(device)
    model = DualTowerContrastive(user_t, prod_t, temperature=args.temperature, eps=1e-8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        iters = 0



        for users, prods in tqdm(loader, desc=f"{cat} epoch {epoch+1}"):
            users = users.to(device); prods = prods.to(device)
            optimizer.zero_grad()
            loss = model(users, prods)
            if not torch.is_tensor(loss):
                loss = torch.tensor(float(loss), requires_grad=True, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().numpy())
            iters += 1
        avg_loss = total_loss / max(iters,1)
        print(f"{cat} epoch {epoch+1} avg loss: {avg_loss:.4f}")
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'model_{cat}.pth')
    torch.save({
        'user_state': user_t.state_dict(),
        'prod_state': prod_t.state_dict(),
        'prod_cols': prod_cols,
        'prod_dim': prod_dim,
        'user_cols': user_cols
    }, save_path)
    print(f"[TRAIN] Saved {save_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--save_dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.7)
    args = parser.parse_args()
    cleaned = auto_clean_data(args.data_dir)
    cust_df = cleaned.get('cust_dataset.csv') if cleaned.get('cust_dataset.csv') is not None else pd.read_csv(os.path.join(args.data_dir,'cust_dataset.csv'))
    events_df = cleaned.get('event_dataset.csv') if cleaned.get('event_dataset.csv') is not None else pd.read_csv(os.path.join(args.data_dir,'event_dataset.csv'))
    prod_df = cleaned.get('product_xlsx') if cleaned.get('product_xlsx') is not None else pd.read_excel(os.path.join(args.data_dir,'productLabels_multiSpreadsheets.xlsx'), sheet_name=None)
    if isinstance(prod_df, dict):
        prod_df = pd.concat(list(prod_df.values()), ignore_index=True, sort=False)
    cust_df = preprocess_users(cust_df)
    cats = ['D','C','A','N','P']
    for c in cats:
        train_for_category(c, args, cust_df, prod_df, events_df)
