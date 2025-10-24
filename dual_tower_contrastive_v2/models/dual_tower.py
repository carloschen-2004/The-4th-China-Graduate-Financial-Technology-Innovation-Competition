import torch
import torch.nn as nn
import torch.nn.functional as F

class DualTowerContrastive(nn.Module):
    def __init__(self, user_tower, prod_tower, temperature=0.5, eps=1e-8):
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

    def embed_users(self, user_x):
        with torch.no_grad():
            emb = self.user_tower(user_x)
            return F.normalize(emb, dim=1, eps=self.eps)

    def embed_products(self, prod_x):
        with torch.no_grad():
            emb = self.prod_tower(prod_x)
            return F.normalize(emb, dim=1, eps=self.eps)
