from __future__ import annotations
import torch
import torch.nn as nn
import os
import numpy as np
import scipy.stats
from collections import deque

def _envf(name, default): 
    try: 
        return float(os.getenv(name, default))
    except: 
        return default

def _envi(name, default):
    try: 
        return int(os.getenv(name, default))
    except: 
        return default

class AACScheduler(nn.Module):
    """
    自适应对抗项权重 w_t:
    w_t = clip( α·Qτ(residual_t) + β·Drift_t , w_min, w_max )
    
    Qτ：最近窗口残差（建议 |O1-W| 的标量化，如按维度取均值后再取高分位）
    Drift：最近残差分布相对历史基线的漂移（建议 JS 散度或简化成 PSI）
    """
    def __init__(
        self,
        tau: float = 0.9,
        alpha: float = 1.0,
        beta: float = 0.5,
        window_size: int = 256,
        w_min: float = 0.0,
        w_max: float = 1.0,
        ema_decay: float = 0.98,
        bins: int = 32,
        eps: float = 1e-6
    ):
        super().__init__()
        # 环境变量覆盖
        tau = _envf("AAC_TAU", tau)
        alpha = _envf("AAC_ALPHA", alpha)
        beta = _envf("AAC_BETA", beta)
        window_size = _envi("AAC_WND", window_size)
        w_min = _envf("AAC_W_MIN", w_min)
        w_max = _envf("AAC_W_MAX", w_max)
        ema_decay = _envf("AAC_EMA", ema_decay)
        bins = _envi("AAC_BINS", bins)
        
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.window_size = window_size
        self.w_min = w_min
        self.w_max = w_max
        self.ema_decay = ema_decay
        self.bins = bins
        self.eps = eps
        
        # 滑动窗口缓冲区
        self.buf = deque(maxlen=window_size)
        self.ref_hist = None
        
        # 便于日志的统计量
        self.last_q = 0.0
        self.last_drift = 0.0
        self.last_w = 0.0

    def update(self, residual_vec: torch.Tensor) -> float:
        """
        residual_vec: (B,) 标量化残差，如 mean(|O1-W|, dim=(1,2))
        返回: 标量 w_t
        """
        v = residual_vec.detach().flatten().cpu().numpy()
        self.buf.extend(v.tolist())
        
        if len(self.buf) < max(16, self.window_size//4):  # 降低预热阈值
            return self.w_min  # 预热期保守
        
        arr = np.array(self.buf)
        q = np.quantile(arr, self.tau)
        
        # 漂移：JS 散度（直方图）
        hist, _ = np.histogram(arr, bins=self.bins, range=(arr.min(), arr.max()), density=True)
        hist = hist + self.eps
        hist = hist / hist.sum()
        
        if self.ref_hist is None:
            self.ref_hist = hist
        else:
            self.ref_hist = self.ema_decay * self.ref_hist + (1-self.ema_decay) * hist
        
        m = 0.5 * (hist + self.ref_hist)
        js = 0.5 * (scipy.stats.entropy(hist, m) + scipy.stats.entropy(self.ref_hist, m))
        
        # 归一 + 裁剪
        q_n = q / (arr.mean() + self.eps)
        js_n = js / (np.log(self.bins))  # JS ∈ [0, ln K]
        w = self.alpha * q_n + self.beta * js_n
        w = float(np.clip(w, self.w_min, self.w_max))
        
        # 更新统计量（便于日志）
        self.last_q = q
        self.last_drift = js
        self.last_w = w
        
        return w
    
    def step(self, residual: torch.Tensor) -> float:
        """
        兼容旧接口：residual: 本 batch 的逐元素残差 (任意形状) 或其向量化
        返回: 标量 w_t
        """
        # 将residual转换为标量化残差
        if residual.dim() > 1:
            # 按维度取均值后再取均值
            residual_scalar = residual.mean(dim=tuple(range(1, residual.dim())))
        else:
            residual_scalar = residual
        
        return self.update(residual_scalar)
    
    def stats(self):
        """返回当前统计信息"""
        return {
            "w": self.last_w,
            "q": self.last_q,
            "drift": self.last_drift,
            "buf_size": len(self.buf),
        }
