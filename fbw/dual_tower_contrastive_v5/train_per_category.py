import os
import argparse
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.user_tower import UserTower
from models.product_tower import ProductTower
from models.dual_tower import DualTowerContrastive
from utils.data_cleaner_v5 import auto_clean_all
from utils.feature_align_v5 import align_product_features,save_aligned

# work_dir: TowTowerNew
current_script_path = os.path.abspath(__file__)
work_dir = os.path.dirname(current_script_path)
# print(work_dir)
LOG_PATH = os.path.join(work_dir,'outputs/train_log.txt')

def ensure_output_dir():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log(s):
    print(s)
    ensure_output_dir()
    with open(LOG_PATH, 'a',encoding='utf-8') as f:
        f.write(str(s)+'\n')

# ---------------- Dataset ----------------
class CategoryDataset(Dataset):
    def __init__(self, merged_df, user_cols, prod_cols, pn='is_success', jitter=1e-6):
        self.merged = merged_df.reset_index(drop=True)
        self.user_cols = user_cols
        self.prod_cols = prod_cols
        self.jitter = jitter
        self.pn = pn # 辨别是否为成功值
    def __len__(self):
        return len(self.merged)
    def __getitem__(self, idx):
        row = self.merged.iloc[idx]
        u = row[self.user_cols].values.astype(float)
        p = row[self.prod_cols].values.astype(float)
        p = p + np.random.normal(scale=self.jitter, size=p.shape)
        pn = int(row[self.pn])
        return u, p, pn

def collate_fn(batch):
    users = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    prods = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    pns = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    return users, prods, pns

# ---------------- Training per category ----------------
def train_for_category(cat, args, cust_df, prod_aligned, event_df):
    cats = event_df['prod_cat'].unique()
    start = time.time()
    log(f"[TRAIN] Starting category {cat}")

    sub = prod_aligned.get(cat)
    if sub is None:
        log(f"[TRAIN] No aligned product data for {cat}")
        return
    prod_sub = sub['df'].reset_index(drop=True)
    prod_cols = sub['numeric_cols']

    # 选择正样本事件(A/B)
    ev_pos = event_df[(event_df['A'] == 1)|(event_df['B'] == 1)].copy()
    # 选择cat相关的事件
    ev_pos = ev_pos[ev_pos['prod_id'].astype(str).isin(prod_sub['prod_id'].astype(str))]
    total_events = len(event_df)
    matched = len(ev_pos)
    log(f"[CHECK] Category {cat}: matched events {matched} / {total_events} ({(matched / total_events * 100) if total_events > 0 else 0:.2f}%)")

    # 使用负样本事件(D)
    if args.use_hard_negative:
        # 选择负样本事件(D)
        ev_neg = event_df[event_df['D'] == 1].copy()
        # 选择cat相关的事件
        ev_neg = ev_neg[ev_neg['prod_id'].astype(str).isin(prod_sub['prod_id'].astype(str))]
        ev = pd.concat([ev_pos, ev_neg], ignore_index=True, sort=False)
    else:
        ev = ev_pos

    merged = ev.merge(cust_df, on='cust_no',how='left').merge(prod_sub, on='prod_id',how='inner')
    if merged.shape[0] == 0:
        log(f"[TRAIN] Merged empty for {cat} after join; possible id mismatch")
        return

    # 用户和产品特征
    cust_feats = [c for c in cust_df.columns if c != 'cust_no']
    user_cols = [c for c in cust_feats if c in merged.columns and merged[c].std() > 0] + [f'prev_count_{cat_name}' for cat_name in cats]
    prod_cols_in_merged = [c for c in prod_cols if c in merged.columns and merged[c].std() > 0]
    print("[用户特征] user_cols:", user_cols)
    print("[产品特征] prod_cols_in_merged:", prod_cols_in_merged)

    if len(user_cols) == 0 or len(prod_cols_in_merged) == 0:
        log(f"[TRAIN] No usable features for {cat}")
        return

    ds = CategoryDataset(merged, user_cols, prod_cols_in_merged, jitter=1e-6)
    if len(ds) < max(16, args.batch_size):
        log(f"[TRAIN] Too few samples ({len(ds)}) for {cat}; skipping.")
        return

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    user_t = UserTower(len(user_cols),embed_dim=args.embed_dim).to(device)
    prod_t = ProductTower(len(prod_cols_in_merged),embed_dim=args.embed_dim).to(device)
    model_params = list(user_t.parameters()) + list(prod_t.parameters())
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    model = DualTowerContrastive(user_t, prod_t, temperature=args.temperature).to(device)


    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        iters = 0
        for users, prods, pns in tqdm(loader, desc=f"{cat} epoch {epoch+1}"):
            users = users.to(device)
            prods = prods.to(device)
            pns = pns.to(device)
            optimizer.zero_grad()

            # 普通对比损失
            loss = model(users, prods)

            # 硬负样本损失
            if args.use_hard_negative:
                mask_pos = (pns == 1)
                u_pos = user_t(users[mask_pos])
                p_pos = prod_t(prods[mask_pos])
                mask_neg = (pns == 0)
                u_neg = user_t(users[mask_neg])
                p_neg = prod_t(prods[mask_neg])
                hn_loss = model.hard_negative_loss(u_pos, p_pos, u_neg, p_neg, margin=args.margin)
                loss = loss + args.neg_weight * hn_loss

            if torch.isnan(loss):
                log(f"[WARN] NaN loss on {cat} epoch {epoch + 1}")
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
    log(f"[TRAIN] Saved {os.path.join(args.save_dir, f'model_{cat}.pth')} time_elapsed={(time.time() - start):.1f}s")

# ---------------- Main ----------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--save_dir', default='outputs')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--use_hard_negative', type=bool, default=True)
    p.add_argument('--margin', type=float, default=0.05)
    p.add_argument('--neg_weight', type=float, default=0.001, dest='neg_weight')
    args = p.parse_args()

    if os.path.exists(LOG_PATH): os.remove(LOG_PATH)
    cust_df, event_df, prod_df = auto_clean_all(args.data_dir)
    aligned = align_product_features(prod_df, min_numeric_cols=4)
    save_aligned(aligned)

    cats = list(aligned.keys())
    for cat in cats:
        #train_for_category(cat, args, cust_df, prod_aligned, event_df)
        train_for_category(cat = cat,
                           args = args,
                           cust_df = cust_df,
                           prod_aligned = aligned,
                           event_df = event_df)

