# -*- coding: utf-8 -*-
from typing import Optional
import torch
import torch.nn as nn

class ITACExportModule(nn.Module):
    """
    仅用于导出（推理）：encoder -> dec1 -> (O1, score)
    - 不包含 GRL/AAC/Phase-2 分支
    - 不返回 dict，避免 TorchScript/ONNX 对象类型限制
    - 只做 'mean' 规约，确保图形稳定
    """
    def __init__(self, encoder: nn.Module, dec1: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.dec1 = dec1
        # 将常量显式声明为 TorchScript Attribute，避免类型检查报错
        self.reduction_id = torch.jit.Attribute(0, int)  # 0==mean

    def forward(self, W: torch.Tensor):
        """
        W: (B, L, D)
        返回:
          O1: (B, L, D)
          score: (B,)  — mean(|O1-W|) over (L,D)
        """
        # 直接调用encoder并解包tuple
        encoder_output = self.encoder(W)
        Z = encoder_output[0]  # 直接取第一个元素（time_feat）
        O1 = self.dec1(Z) + W  # 残差重构
        rec = (O1 - W).abs()
        # 只走 mean，保证可脚本化
        score = rec.mean(dim=(1, 2))
        return O1, score


def make_exportable(itac_ad_module: nn.Module) -> ITACExportModule:
    """
    从你的 iTAC-AD 实例中提取 encoder/dec1，构造可导出的瘦身模块。
    """
    enc = getattr(itac_ad_module, "encoder", None)
    dec1 = getattr(itac_ad_module, "dec1", None)
    if enc is None or dec1 is None:
        raise RuntimeError("ITAC-AD 模型缺少 encoder/dec1，无法构造导出模块。")
    exp = ITACExportModule(enc, dec1)
    exp.eval()
    return exp
