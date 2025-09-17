# itac_ad/core/grl.py
from __future__ import annotations
import torch
from torch.autograd import Function
import torch.nn as nn

class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None

class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
    def forward(self, x):
        return _GRL.apply(x, self.lambd)

class WarmupGRL(GradientReversal):
    def __init__(self, lambd: float = 1.0, warmup_steps: int = 200):
        super().__init__(lambd)
        self.max_lambd = float(lambd)
        self.cur = 0
        self.warm = int(warmup_steps)
    
    def step(self):  # 训练每个 batch 调用一次
        if self.warm <= 0: 
            return
        self.cur += 1
        ratio = min(1.0, self.cur / max(1, self.warm))
        self.lambd = self.max_lambd * ratio
