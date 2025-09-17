from __future__ import annotations
import os
import torch
import torch.nn as nn
from itac_ad.components.itr_encoder import VariateTokenEncoder
from itac_ad.components.itransformer_backbone import iTransformerBackbone
from itac_ad.components.tranad_decoders import build_tranad_decoders
from itac_ad.core.grl import GradientReversal, WarmupGRL

def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except:
        return default
def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except:
        return default
def _env_str(name, default):
    return str(os.getenv(name, default))

class ITAC_AD(nn.Module):
    """
    Encoder(变量token) + 双解码器 (phase1/phase2)
    编码器可选:
      - encoder_kind = 'itr'        : iTransformer 风格编码器（默认）
      - encoder_kind = 'itransformer': 完整 iTransformer backbone
    解码器可选:
      - decoder_kind = 'mlp'   : 两层 MLP 残差（最简）
      - decoder_kind = 'tranad': 优先 vendor 的 TranAD 解码器，fallback 时间域 Transformer
    """
    def __init__(
        self,
        feats: int,  # TranAD 兼容性：特征维度
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        dropout: float = 0.1,
        lazy_build: bool = True,
        encoder_kind: str = None,
        decoder_kind: str = None,
    ):
        super().__init__()
        # TranAD 兼容性属性
        self.name = 'ITAC_AD'
        self.lr = 0.0001
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        
        # 覆盖默认值（可通过环境变量）
        d_model  = _env_int("ITAC_D_MODEL", d_model)
        n_heads  = _env_int("ITAC_N_HEADS", n_heads)
        e_layers = _env_int("ITAC_E_LAYERS", e_layers)
        dropout  = _env_float("ITAC_DROPOUT", dropout)
        encoder_kind = _env_str("ITAC_ENCODER", encoder_kind or "itr").lower()
        decoder_kind = _env_str("ITAC_DECODER", decoder_kind or "tranad").lower()

        # GRL 对抗训练配置
        use_grl  = os.getenv("ITAC_USE_GRL", "1") not in ("0","false","False")
        grl_lamb = _env_float("ITAC_GRL_LAMBDA", 1.0)
        warm_steps = _env_int("ITAC_GRL_WARMUP", 200)
        self.uses_grl = bool(use_grl)
        self.grl = WarmupGRL(grl_lamb, warm_steps) if self.uses_grl else nn.Identity()

        # 选择编码器
        if encoder_kind == "itransformer":
            # 使用完整的iTransformer backbone
            self.encoder = None  # 将在forward中lazy构建
            self.encoder_kind = "itransformer"
        else:
            # 使用iTransformer风格的编码器（默认）
            self.encoder = VariateTokenEncoder(
                d_model=d_model, n_heads=n_heads, e_layers=e_layers, dropout=dropout
            )
            self.encoder_kind = "itr"
        
        self.lazy_build = lazy_build
        self.decoder_kind = decoder_kind
        self.dec1 = None
        self.dec2 = None

    def _build_decoders(self, D: int):
        if self.decoder_kind == "mlp":
            def make_dec():
                return nn.Sequential(
                    nn.Linear(D, D),
                    nn.GELU(),
                    nn.Linear(D, D)
                )
            self.dec1 = make_dec()
            self.dec2 = make_dec()
        else:
            # TranAD 风格（优先 vendor）
            self.dec1, self.dec2 = build_tranad_decoders(
                input_dim=D,
                depth=_env_int("ITAC_DEC_DEPTH", 2),
                d_model=_env_int("ITAC_DEC_DMODEL", None) or None,
                n_heads=_env_int("ITAC_DEC_HEADS", 4),
                dropout=_env_float("ITAC_DEC_DROPOUT", 0.1),
            )

    def forward(self, x: torch.Tensor, *, phase: int = 2, aac_w: float = 0.0):
        """
        严格对齐 TranAD Phase-2 的对抗式重构梯度方向
        使用自条件（self-conditioning）而不是GRL
        
        Args:
            x: 输入 [B,T,D]
            phase: 1=仅D1重构, 2=D1+D2+自条件对抗
            aac_w: AAC权重，用于Phase-2的对抗损失
            
        Returns:
            dict with loss, O1, O2, loss_rec, loss_adv
        """
        assert x.dim() == 3, f"ITAC_AD expects [B,T,D], got {tuple(x.shape)}"
        
        B, T, D = x.shape
        
        # 编码器处理
        if self.encoder_kind == "itransformer":
            # 使用完整的iTransformer backbone
            if self.encoder is None:
                self.encoder = iTransformerBackbone(
                    d_in=D,
                    seq_len=T,
                    d_model=_env_int("ITAC_D_MODEL", 128),
                    n_heads=_env_int("ITAC_N_HEADS", 8),
                    e_layers=_env_int("ITAC_E_LAYERS", 2),
                    dropout=_env_float("ITAC_DROPOUT", 0.1)
                )
            h_time = self.encoder(x)  # [B,T,D]
            h_tok = None  # iTransformer backbone不返回token特征
        else:
            # 使用iTransformer风格的编码器
            h_time, h_tok = self.encoder(x)  # [B,T,D], [B,D,C]
        
        if (self.dec1 is None or self.dec2 is None):
            self._build_decoders(D)

        # Phase-1: 仅D1重构（无异常分数）
        o1 = self.dec1(h_time) + x              # 残差重构
        
        if phase == 1:
            # Phase-1: 仅重构损失（使用SmoothL1更鲁棒）
            loss_rec = torch.nn.functional.smooth_l1_loss(o1, x, reduction="mean", beta=1.0)
            return {
                "loss": loss_rec,
                "O1": o1,
                "loss_rec": loss_rec,
                "loss_adv": torch.tensor(0.0, device=x.device),
                "O2": torch.zeros_like(o1)
            }
        else:
            # Phase-2: 使用自条件（TranAD风格）
            # 计算Phase-1的重构误差作为条件
            c = (o1 - x) ** 2  # 异常分数条件
            
            # 将条件与原始输入拼接，重新编码
            x_cond = torch.cat([x, c], dim=-1)  # [B,T,2D]
            
            # 重新编码带条件的输入
            if self.encoder_kind == "itransformer":
                # 为iTransformer backbone重新构建编码器处理2D输入
                if not hasattr(self, 'encoder_2d') or self.encoder_2d is None:
                    self.encoder_2d = iTransformerBackbone(
                        d_in=2*D,
                        seq_len=T,
                        d_model=_env_int("ITAC_D_MODEL", 128),
                        n_heads=_env_int("ITAC_N_HEADS", 8),
                        e_layers=_env_int("ITAC_E_LAYERS", 2),
                        dropout=_env_float("ITAC_DROPOUT", 0.1)
                    )
                h_cond = self.encoder_2d(x_cond)  # [B,T,2D]
            else:
                h_cond, _ = self.encoder(x_cond)   # [B,T,2D]
            
            # Phase-2: 使用条件编码进行重构
            # 注意：h_cond现在是[B,T,2D]，但decoder期望[B,T,D]
            # 我们需要将h_cond投影回D维
            if h_cond.shape[-1] != D:
                # 如果decoder还没有为2D输入构建，重新构建
                if not hasattr(self, 'dec2_2d') or self.dec2_2d is None:
                    self.dec2_2d, _ = build_tranad_decoders(
                        input_dim=h_cond.shape[-1],
                        depth=_env_int("ITAC_DEC_DEPTH", 2),
                        d_model=_env_int("ITAC_DEC_DMODEL", None) or None,
                        n_heads=_env_int("ITAC_DEC_HEADS", 4),
                        dropout=_env_float("ITAC_DEC_DROPOUT", 0.1),
                    )
                o2_raw = self.dec2_2d(h_cond)  # [B,T,2D]
                # 将2D输出投影回D维
                o2 = o2_raw[..., :D] + x  # 只取前D维
            else:
                o2 = self.dec2(h_cond) + x
            
            loss_rec = torch.nn.functional.smooth_l1_loss(o1, x, reduction="mean", beta=1.0)
            loss_adv = torch.nn.functional.smooth_l1_loss(o2, x, reduction="mean", beta=1.0)
            loss = loss_rec + aac_w * loss_adv
            
            return {
                "loss": loss,
                "O1": o1,
                "O2": o2,
                "loss_rec": loss_rec,
                "loss_adv": loss_adv,
                "h_token": h_tok
            }
    
    def export_sanity(self, L:int=16, D:int=8, device:str="cpu"):
        """快速自检导出路径是否纯张量"""
        import torch
        from itacad.exportable import make_exportable
        self.eval().to(device)
        exp = make_exportable(self).to(device)
        W = torch.randn(2, L, D, device=device)
        O1, score = exp(W)
        assert O1.shape == W.shape and score.shape == (2,)
        return True