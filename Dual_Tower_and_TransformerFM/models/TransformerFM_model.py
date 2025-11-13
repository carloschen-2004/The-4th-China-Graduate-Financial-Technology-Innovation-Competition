import torch
import torch.nn as nn
import torch.nn.functional as F

# ============= 定义Factorization Machine (FM)组件 =============
class FM(nn.Module):
    """Factorization Machine - 捕捉特征间的二阶交互"""

    def __init__(self, num_features, k):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)  # 线性部分
        self.V = nn.Parameter(torch.randn(num_features, k))  # 隐向量矩阵
        # 初始化
        nn.init.normal_(self.V, mean=0, std=0.01)

    def forward(self, x):
        # x shape: [batch, num_features]
        # 线性部分
        linear_part = self.linear(x)  # [batch, 1]

        # 交互部分: 使用优化的FM公式
        # sum((x*V)^2 - (x^2)*(V^2)) / 2
        inter_1 = torch.pow(torch.mm(x, self.V), 2)  # [batch, k]
        inter_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))  # [batch, k]
        interaction_part = 0.5 * torch.sum(inter_1 - inter_2, dim=1, keepdim=True)  # [batch, 1]

        return linear_part + interaction_part


# ============= 定义融合Transformer和FM的模型 =============
class ProductTransformerFM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.2, num_heads=4, fm_k=16,
                 fusion='concat'):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: Transformer隐藏维度
            num_classes: 输出类别数
            num_layers: Transformer层数
            dropout: Dropout率
            num_heads: 注意力头数
            fm_k: FM隐向量维度
            fusion: 融合方式 ('concat' 或 'add')
        """
        super(ProductTransformerFM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fusion = fusion

        # ========== Transformer部分 ==========
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, hidden_dim))
        # 输入投影层
        self.input_proj = nn.Linear(1, hidden_dim)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # Transformer Encoder层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ========== FM部分 ==========
        self.fm = FM(num_features=input_dim, k=fm_k)

        # ========== 融合层 ==========
        if fusion == 'concat':
            # 将Transformer输出和FM输出concat
            fusion_dim = hidden_dim + 1  # FM输出是标量
            self.fusion_proj = nn.Linear(fusion_dim, hidden_dim)
        else:  # fusion == 'add'
            # FM输出投影到hidden_dim后相加
            self.fm_proj = nn.Linear(1, hidden_dim)
            fusion_dim = hidden_dim

        # ========== 输出层 ==========
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: [batch, input_dim]
        batch_size = x.size(0)

        # ========== Transformer路径 ==========
        # Reshape成序列: [batch, seq_len, 1]
        x_reshaped = x.unsqueeze(2)  # [batch, input_dim, 1]
        # 输入投影
        x_proj = self.input_proj(x_reshaped)  # [batch, input_dim, hidden_dim]
        # 添加位置编码
        x_proj = x_proj + self.pos_embedding  # [batch, input_dim, hidden_dim]
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        x_proj = torch.cat([cls_tokens, x_proj], dim=1)  # [batch, input_dim+1, hidden_dim]
        # Transformer编码
        x_transformed = self.transformer(x_proj)  # [batch, input_dim+1, hidden_dim]
        # 使用CLS token的输出作为序列表示
        transformer_output = x_transformed[:, 0, :]  # [batch, hidden_dim]

        # ========== FM路径 ==========
        fm_output = self.fm(x)  # [batch, 1]

        # ========== 融合两部分 ==========
        if self.fusion == 'concat':
            # 将FM输出和Transformer输出concat
            fused = torch.cat([transformer_output, fm_output], dim=1)  # [batch, hidden_dim + 1]
            fused = self.fusion_proj(fused)  # [batch, hidden_dim]
        else:  # fusion == 'add'
            # 将FM输出投影到hidden_dim后相加
            fm_proj = self.fm_proj(fm_output)  # [batch, hidden_dim]
            fused = transformer_output + fm_proj  # [batch, hidden_dim]

        # ========== 输出层 ==========
        out = self.fc1(fused)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out