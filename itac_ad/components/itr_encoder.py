# itac_ad/components/itr_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn

class VariateTokenEncoder(nn.Module):
    """
    输入:  x [B, T, D]
    输出:  h_time [B, T, D] (与输入同形状，用于解码/重构)
          h_token [B, D, d_model] (可选: 供上层使用)
    关键点: 把每个变量的时间序列长度 T 投影到 d_model，变量数 D 当作“序列长度”送入 TransformerEncoder。
    """
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        dropout: float = 0.1,
        act: str = "gelu",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.dropout = dropout
        self.act = act

        # lazy: 首次 forward 时根据 T 构建投影层
        self.token_proj = None       # Linear(T -> d_model)
        self.time_reproj = None      # Linear(d_model -> T)
        self.encoder = None          # nn.TransformerEncoder(batch_first=True)
        self.norm = None

    def _build(self, T: int):
        act = nn.ReLU() if self.act == "relu" else nn.GELU()
        self.token_proj = nn.Sequential(
            nn.LayerNorm((T,)),
            nn.Linear(T, self.d_model),
        )
        self.time_reproj = nn.Sequential(
            nn.Linear(self.d_model, T),
            nn.LayerNorm((T,)),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,   # 允许 [B, S, E]
            activation="gelu" if self.act != "relu" else "relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.e_layers)
        self.norm = nn.LayerNorm(self.d_model)
        
        # 确保所有参数都是 float 类型并移动到正确设备
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        self.token_proj = self.token_proj.float().to(device)
        self.time_reproj = self.time_reproj.float().to(device)
        self.encoder = self.encoder.float().to(device)
        self.norm = self.norm.float().to(device)

    def forward(self, x: torch.Tensor):
        # x: [B, T, D] 或 [B, T, 2D] (Phase-2时)
        B, T, D = x.shape
        if self.token_proj is None:
            self._build(T)

        # 变量作 token: [B, T, D] -> [B, D, T]
        v = x.transpose(1, 2).contiguous().view(B * D, T)
        tokens = self.token_proj(v).view(B, D, self.d_model)  # [B, D, d_model]

        # Transformer 在 D 维处理
        h = self.encoder(tokens)                               # [B, D, d_model]
        h = self.norm(h)

        # 映回时间维: [B, D, d_model] -> [B, D, T] -> [B, T, D]
        time_feat = self.time_reproj(h).view(B, D, T).transpose(1, 2).contiguous()

        return time_feat, h
