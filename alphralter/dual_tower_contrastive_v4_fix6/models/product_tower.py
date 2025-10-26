import torch.nn as nn
class ProductTower(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super().__init__()
        if input_dim <= 0:
            self.net = nn.Sequential(nn.Linear(1, embed_dim))
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, embed_dim)
            )
    def forward(self, x):
        if x.shape[1] == 0:
            import torch
            x = torch.zeros((x.shape[0],1), device=x.device, dtype=x.dtype)
        return self.net(x)
