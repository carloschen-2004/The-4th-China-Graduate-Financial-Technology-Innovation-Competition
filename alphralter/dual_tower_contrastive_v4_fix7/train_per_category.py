import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import normalize
from tqdm import tqdm

from models.user_tower import UserTower
from models.product_tower import ProductTower
from models.dual_tower import DualTowerContrastive
from utils.data_cleaner_v4 import auto_clean_all
from utils.feature_align import align_product_features, save_aligned
from utils.preprocess import preprocess_users_cust_all

LOG_PATH = 'outputs/train_log.txt'

def ensure_output_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log(s):
    print(s)
    ensure_output_dir()
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(str(s) + '\n')

# ---------------- Dataset ----------------
class CategoryDataset(Dataset):
    def __init__(self, merged_df, user_cols, prod_cols, jitter=1e-6):
        self.merged = merged_df.reset_index(drop=True)
        self.user_cols = user_cols
        self.prod_cols = prod_cols
        self.jitter = jitter
        if len(self.prod_cols) == 0:
            self.merged['prod_hash'] = self.merged['prod_id'].astype(str).apply(lambda x: hash(x)%1000)
            self.prod_cols = ['prod_hash']
    def __len__(self):
        return len(self.merged)
    def __getitem__(self, idx):
        row = self.merged.iloc[idx]
        u = row[self.user_cols].values.astype(float)
        p = row[self.prod_cols].values.astype(float) if len(self.prod_cols)>0 else np.array([0.0])
        p = p + np.random.normal(scale=self.jitter, size=p.shape)
        return u, p

def collate_fn(batch):
    users = np.stack([b[0] for b in batch])
    prods = np.stack([b[1] for b in batch])
    users = torch.tensor(users, dtype=torch.float32)
    prods = torch.tensor(prods, dtype=torch.float32)
    return users, prods

# ---------------- Training per category ----------------
def train_for_category(cat, args, cust_df, prod_aligned, events_df):
    start = time.time()
    log(f"[TRAIN] Starting category {cat}")

    sub = prod_aligned.get(cat)
    if sub is None:
        log(f"[TRAIN] No aligned product data for {cat}")
        return

    prod_sub = sub['df'].reset_index(drop=True)
    prod_cols = sub['numeric_cols']

    # 选择正样本事件 (A/B)
    ev_pos = events_df[(events_df['A'] == 1) | (events_df['B'] == 1)].copy()
    ev_pos = ev_pos[ev_pos['prod_id'].astype(str).isin(prod_sub['prod_id'].astype(str))]
    total_events = len(events_df)
    matched = len(ev_pos)
    log(f"[CHECK] Category {cat}: matched events {matched} / {total_events} ({(matched/total_events*100) if total_events>0 else 0:.2f}%)")
    if ev_pos.shape[0] == 0:
        log(f"[TRAIN] No positive events for {cat}")
        return

    # 如果使用负样本事件（D）
    if args.use_hard_negative:
        ev_neg = events_df[events_df['D'] == 1].copy()
        ev_neg = ev_neg[ev_neg['prod_id'].astype(str).isin(prod_sub['prod_id'].astype(str))]
        ev = pd.concat([ev_pos, ev_neg], ignore_index=True, sort=False)
    else:
        ev = ev_pos

    merged = ev.merge(cust_df, on='cust_no', how='left').merge(prod_sub, on='prod_id', how='inner')
    if merged.shape[0] == 0:
        log(f"[TRAIN] Merged empty for {cat} after join; possible id mismatch")
        return

    # 用户和产品特征
    cust_feats = [c for c in cust_df.columns if c != 'cust_no']
    user_cols = [c for c in cust_feats if c in merged.columns and merged[c].std() > 0]
    prod_cols_in_merged = [c for c in prod_cols if c in merged.columns and merged[c].std() > 0]

    if len(user_cols) == 0 or len(prod_cols_in_merged) == 0:
        log(f"[TRAIN] No usable features for {cat}")
        return

    ds = CategoryDataset(merged, user_cols, prod_cols_in_merged, jitter=1e-6)
    if len(ds) < max(16, args.batch_size):
        log(f"[TRAIN] Too few samples ({len(ds)}) for {cat}; skipping.")
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    user_t = UserTower(len(user_cols), embed_dim=args.embed_dim).to(device)
    prod_t = ProductTower(len(prod_cols_in_merged), embed_dim=args.embed_dim).to(device)
    model_params = list(user_t.parameters()) + list(prod_t.parameters())
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    model = DualTowerContrastive(user_t, prod_t, temperature=args.temperature).to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        iters = 0
        for users, prods in tqdm(loader, desc=f"{cat} epoch {epoch+1}"):
            users = users.to(device)
            prods = prods.to(device)
            optimizer.zero_grad()

            # 普通对比损失
            loss = model(users, prods)

            # 硬负样本损失
            if args.use_hard_negative:
                u_pos = user_t(users)
                p_pos = prod_t(prods)
                # 简单示例：随机打乱 batch 生成负样本
                perm = torch.randperm(u_pos.size(0))
                u_neg = u_pos[perm]
                p_neg = p_pos[perm]
                hn_loss = model.hard_negative_loss(u_pos, p_pos, u_neg, p_neg, margin=args.margin)
                loss = loss + args.neg_weight * hn_loss

            if torch.isnan(loss):
                log(f"[WARN] NaN loss on {cat} epoch {epoch+1}")
                loss = torch.tensor(0.0, requires_grad=True, device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().numpy())
            iters += 1

        avg_loss = total_loss / max(iters, 1)
        log(f"{cat} epoch {epoch+1} avg loss: {avg_loss:.6f}")

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'user_state': user_t.state_dict(),
        'prod_state': prod_t.state_dict(),
        'prod_cols': prod_cols_in_merged,
        'prod_dim': len(prod_cols_in_merged),
        'user_cols': user_cols
    }, os.path.join(args.save_dir, f'model_{cat}.pth'))
    log(f"[TRAIN] Saved {os.path.join(args.save_dir, f'model_{cat}.pth')} time_elapsed={(time.time()-start):.1f}s")



# ---------------- Main ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--save_dir', default='outputs')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--use_hard_negative', type=bool, default=False)
    p.add_argument('--margin', type=float, default=0.05)
    p.add_argument('--neg_weight', type=float, default=0.001, dest='neg_weight')
    args = p.parse_args()

    if os.path.exists(LOG_PATH): os.remove(LOG_PATH)

    cust_df, events_df, prod_df = auto_clean_all(args.data_dir)
    aligned = align_product_features(prod_df, min_numeric_cols=4)
    save_aligned(args.data_dir, aligned)

    cust_df, _ = preprocess_users_cust_all(cust_df)
    os.makedirs(os.path.join(args.data_dir,'aligned'), exist_ok=True)
    cust_df.to_csv(os.path.join(args.data_dir,'aligned','aligned_cust_dataset.csv'), index=False)

    cats = list(aligned.keys())
    for c in cats:
        train_for_category(c, args, cust_df, aligned, events_df)
