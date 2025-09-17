# itac_ad/components/tranad_decoders.py
from __future__ import annotations
import importlib
import torch
import torch.nn as nn

def _try_vendor_decoders():
    """
    动态尝试从 vendor/tranad/src/models.py 里拿到两路 decoder。
    你本地如果类名不同（比如 Reconstruction/Adversarial），也能兼容。
    返回: callables (dec1_ctor, dec2_ctor) 或 (None, None)
    """
    try:
        m = importlib.import_module("vendor.tranad.src.models")
    except Exception:
        return None, None

    # 常见命名猜测表：你可以按需扩展
    candidates = [
        ("Decoder1", "Decoder2"),
        ("ReconstructionDecoder", "AdversarialDecoder"),
        ("Decoder", "DecoderAdv"),
        ("Dec1", "Dec2"),
    ]
    for n1, n2 in candidates:
        if hasattr(m, n1) and hasattr(m, n2):
            return getattr(m, n1), getattr(m, n2)

    # 退而求其次：如果只有一个 Decoder，就复用两次
    for name in ["Decoder", "TranADDecoder", "ReconstructionDecoder"]:
        if hasattr(m, name):
            ctor = getattr(m, name)
            return ctor, ctor

    return None, None


class _TimeTransformerBlock(nn.Module):
    """
    稳定的"TranAD 风格"时间域块：
    对 [B,T,D] 在 T 维做 TransformerEncoder，再做特征维 MLP 残差。
    """
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=1)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x):  # [B,T,D]
        h = self.enc(x)
        h = self.ln(h)
        return h + self.ff(h)


class TranADStyleDecoder(nn.Module):
    """
    安全的 fallback 解码器：堆叠几个时间域块，最后 1x1 线性映回 D。
    与输入同形 [B,T,D] -> [B,T,D]
    """
    def __init__(self, D: int, depth: int = 2, d_model: int = None, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        d_model = d_model or max(64, ((D + 7) // 8) * 8)  # 稳健取值
        self.inp = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([_TimeTransformerBlock(d_model, n_heads, ff_mult=4, dropout=dropout) for _ in range(depth)])
        self.out = nn.Linear(d_model, D)

    def forward(self, x):  # [B,T,D]
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        y = self.out(h)
        return y


def build_tranad_decoders(input_dim: int, *, depth: int = 2, d_model: int | None = None,
                          n_heads: int = 4, dropout: float = 0.1):
    """
    返回两路解码器 (dec1, dec2)。优先 vendor，fallback 为 TranADStyleDecoder。
    """
    v1, v2 = _try_vendor_decoders()
    if v1 is not None and v2 is not None:
        try:
            # 常见构造的适配：有的 vendor 解码器只需要 D，有的还要内隐超参
            dec1 = v1(input_dim) if callable(v1) else v1
            dec2 = v2(input_dim) if callable(v2) else v2
            # 快速 shape 预检（不真正运行）
            assert isinstance(dec1, nn.Module) and isinstance(dec2, nn.Module)
            return dec1, dec2
        except Exception:
            pass  # 回退

    # fallback：稳定可用
    dec1 = TranADStyleDecoder(D=input_dim, depth=depth, d_model=d_model, n_heads=n_heads, dropout=dropout)
    dec2 = TranADStyleDecoder(D=input_dim, depth=depth, d_model=d_model, n_heads=n_heads, dropout=dropout)
    return dec1, dec2
