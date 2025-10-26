import torch, torch.nn as nn, torch.nn.functional as F

class DualTowerContrastive(nn.Module):
    def __init__(self, user_tower, prod_tower, temperature=1.0, eps=1e-8):
        super().__init__()
        self.user_tower = user_tower
        self.prod_tower = prod_tower
        self.temperature = temperature
        self.eps = eps

    def forward(self, user_x, prod_x):
        u = self.user_tower(user_x)
        p = self.prod_tower(prod_x)
        u = F.normalize(u, dim=1, eps=self.eps)
        p = F.normalize(p, dim=1, eps=self.eps)
        logits = torch.matmul(u, p.T) / (self.temperature + self.eps)
        logits = torch.clamp(logits, -50.0, 50.0)
        labels = torch.arange(u.size(0), device=u.device)
        loss = F.cross_entropy(logits, labels)
        if torch.isnan(loss):
            return torch.tensor(0.0, requires_grad=True, device=u.device)
        return loss
    def hard_negative_loss(self, u_pos, p_pos, u_neg, p_neg, margin=0.2):
        if u_neg is None or p_neg is None or u_neg.shape[0]==0:
            return torch.tensor(0.0, device=u_pos.device)
        sim_pos = (u_pos * p_pos).sum(dim=1)
        sim_neg = (u_neg * p_neg).sum(dim=1)
        pos_mean = sim_pos.mean()
        neg_mean = sim_neg.mean()
        loss = torch.clamp(margin - pos_mean + neg_mean, min=0.0)
        return loss
    def embed_users(self, user_x):
        with torch.no_grad():
            emb = self.user_tower(user_x)
            return F.normalize(emb, dim=1, eps=self.eps)

    def embed_products(self, prod_x):
        with torch.no_grad():
            emb = self.prod_tower(prod_x)
            return F.normalize(emb, dim=1, eps=self.eps)
