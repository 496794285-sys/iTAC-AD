# -*- coding: utf-8 -*-
import numpy as np, torch

def _to_torch(x, device="cpu"): 
    """将numpy数组转换为torch张量"""
    return torch.as_tensor(x, dtype=torch.float32, device=device)

@torch.no_grad()
def score_windows(model, W_np, mu=None, sigma=None, agg="topk_mean", topk_ratio=0.1, device=None):
    """
    对窗口进行评分，支持通道稳健化和Top-K聚合
    
    Args:
        model: 训练好的模型
        W_np: 窗口数据 (N,L,D) numpy数组
        mu: 训练集均值，用于通道稳健归一化
        sigma: 训练集标准差，用于通道稳健归一化
        agg: 聚合方式 "mean" | "p95" | "topk_mean"
        topk_ratio: Top-K比例，取前 K=ceil(ratio*D) 个通道的平均
        device: 计算设备
    
    Returns:
        (N,) 每个窗口的分数
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    W = _to_torch(W_np, device)
    
    # 通道稳健归一化
    if mu is not None and sigma is not None:
        mu_t = _to_torch(mu.reshape(1,1,-1), device)
        sg_t = _to_torch(sigma.reshape(1,1,-1), device)
        Wn = (W - mu_t) / sg_t
    else:
        Wn = W

    # 模型前向传播
    out = model(Wn, phase=1, aac_w=0.0) if "phase" in model.forward.__code__.co_varnames else model(Wn)
    O1 = out["O1"] if isinstance(out, dict) and "O1" in out else out
    
    # 计算重构残差
    R = (O1 - Wn).abs()                         # (N,L,D)
    # 先沿时间维度均值到 (N,D)，再做通道聚合更稳定
    Rch = R.mean(dim=1)                         # (N,D)

    # 通道聚合策略
    if agg == "p95":
        sc = torch.quantile(Rch, 0.95, dim=1)
    elif agg == "topk_mean":
        N, D = Rch.shape
        k = max(1, int(np.ceil(D * float(topk_ratio))))
        topk = torch.topk(Rch, k=k, dim=1).values
        sc = topk.mean(dim=1)
    else:  # "mean"
        sc = Rch.mean(dim=1)

    return sc.detach().cpu().numpy()

def make_windows(X, L=100, stride=1):
    """
    将时间序列数据转换为滑动窗口
    
    Args:
        X: 时间序列数据 (T,D)
        L: 窗口长度
        stride: 步长
    
    Returns:
        窗口数据 (N,L,D)
    """
    T, D = X.shape
    windows = []
    
    for i in range(0, T - L + 1, stride):
        windows.append(X[i:i+L])
    
    return np.array(windows)

def aggregate_scores(window_scores, window_size, stride, total_length):
    """
    将窗口分数聚合回时间序列分数
    
    Args:
        window_scores: 窗口分数 (N,)
        window_size: 窗口大小
        stride: 步长
        total_length: 总时间长度
    
    Returns:
        时间序列分数 (T,)
    """
    T = total_length
    scores = np.zeros(T)
    counts = np.zeros(T)
    
    for i, score in enumerate(window_scores):
        start = i * stride
        end = min(start + window_size, T)
        # 窗口中心投票
        center = (start + end) // 2
        if center < T:
            scores[center] += score
            counts[center] += 1
    
    # 避免除零
    scores = np.where(counts > 0, scores / counts, 0)
    return scores
