# -*- coding: utf-8 -*-
import os, json, time, math
from typing import Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError:
    yaml = None

# 你已有的事件级指标与 POT
from itac_eval.metrics import pot_threshold, binarize_by_threshold, event_f1, pr_auc
from itacad.thresholding import load_threshold, fit_threshold
from itacad.score import score_windows, aggregate_scores

def _discover_ckpt(ckpt_dir: str) -> str:
    cand = []
    for fn in os.listdir(ckpt_dir):
        if fn.endswith((".pt",".pth",".ckpt")):
            cand.append(os.path.join(ckpt_dir, fn))
    if not cand:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def _load_config(ckpt_dir: str) -> Dict:
    if yaml is None:
        return {}
    for name in ("config.yaml","config.yml","hparams.yaml"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            with open(p, "r") as f:
                return yaml.safe_load(f)
    return {}

def _maybe_load_stats(ckpt_dir):
    """优先加载训练集统计，用于归一化"""
    import json, os
    p = os.path.join(ckpt_dir, "train_stats.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return np.asarray(d["mean"], float), np.asarray(d["std"], float)
    return None, None

def load_model(ckpt_dir: str, device: Optional[str] = None, feats: int = 7):
    """
    约定：你的训练已保存 state_dict 到 ckpt；本函数加载 ITAC_AD。
    """
    from itac_ad.models.itac_ad import ITAC_AD  # 按你的路径修改
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = _load_config(ckpt_dir)
    model = ITAC_AD(feats=feats, **cfg.get("model", {})) if "model" in cfg else ITAC_AD(feats=feats)
    ckpt_path = _discover_ckpt(ckpt_dir)
    sd = torch.load(ckpt_path, map_location="cpu")
    # 兼容多种保存格式: {"state_dict": ...}, {"model": ...}, 或直接 state_dict
    if "state_dict" in sd:
        state_dict = sd["state_dict"]
    elif "model" in sd:
        state_dict = sd["model"]
    else:
        state_dict = sd
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    return model, device, cfg

def _sliding_windows(X: np.ndarray, L: int, stride: int) -> np.ndarray:
    """
    X: (T, D) -> (N, L, D) 以 stride 滑窗，不足 L 的尾部丢弃
    """
    T, D = X.shape
    if T < L:
        return np.empty((0, L, D), dtype=X.dtype)
    idx = np.arange(0, T - L + 1, stride)
    out = np.stack([X[i:i+L] for i in idx], axis=0)
    return out

def _aggregate_scores(win_scores: np.ndarray, L: int, stride: int, T: int) -> np.ndarray:
    """
    把每窗一个分数，映射回时间轴（简单：窗中心点对齐）。
    """
    if len(win_scores)==0:
        return np.zeros(T, dtype=float)
    centers = np.arange(0, T - L + 1, stride) + (L//2)
    s = np.zeros(T, dtype=float); c = np.zeros(T, dtype=float)
    for sc, t in zip(win_scores, centers):
        s[t] += float(sc); c[t] += 1.0
    c[c==0] = 1.0
    return s / c

@torch.no_grad()
def predict_csv(
    csv_path: str,
    ckpt_dir: str,
    window: int,
    stride: int = 1,
    normalize: str = "zscore",
    pot_q: float = 0.98,
    pot_level: float = 0.99,
    event_iou: float = 0.1,
    label_col: Optional[str] = None,
    out_dir: Optional[str] = None,
    score_reduction: str = "mean"
) -> Dict:
    """
    对单个 CSV 做推理。CSV 要求：每列为一个变量，若有标签列，通过 label_col 指定（0/1）。
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if label_col and label_col in df.columns:
        labels = df[label_col].to_numpy().astype(int)
        X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    else:
        labels = None
        X = df.to_numpy(dtype=np.float32)
    T, D = X.shape

    # 归一化：优先使用训练集统计
    mu0, sigma0 = _maybe_load_stats(ckpt_dir)
    if mu0 is not None:
        mu = mu0.reshape(1,-1); sigma = sigma0.reshape(1,-1)
        Xn = (X - mu) / sigma
        print(f"[info] 使用训练集统计进行归一化")
    else:
        # 兜底：仍支持 zscore on test，但打印告警
        print("[warn] train_stats.json not found; normalize on test CSV")
        if normalize == "zscore":
            mu, sigma = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)+1e-8
            Xn = (X - mu)/sigma
        elif normalize == "minmax":
            mn, mx = X.min(axis=0, keepdims=True), X.max(axis=0, keepdims=True)
            Xn = (X - mn)/(mx-mn+1e-8)
        else:
            Xn = X

    # 滑窗
    W = _sliding_windows(Xn, window, stride)  # (N,L,D)
    model, device, cfg = load_model(ckpt_dir, device="cpu", feats=D)  # 强制使用CPU

    # 使用新的评分方法（通道稳健化 + Top-K聚合）
    agg_method = os.getenv("ITAC_SCORE_AGG", "topk_mean")
    topk_ratio = float(os.getenv("ITAC_TOPK", "0.1"))
    
    win_scores = score_windows(
        model, W, mu0, sigma0,
        agg=agg_method,
        topk_ratio=topk_ratio,
        device="cpu"
    )

    scores = aggregate_scores(win_scores, window, stride, T)  # (T,)
    
    # 优先读取训练时标定的阈值
    thr, thr_meta = load_threshold(ckpt_dir)
    if thr is None:
        # 无保存则回退：用测试 scores 估计（fit_threshold）
        print("[warn] threshold.json not found; estimating threshold from test scores")
        thr, _ = fit_threshold(scores, q=pot_q, level=pot_level, method="auto")
    else:
        print(f"[info] 使用训练阈值: {thr:.4f}")
    
    pred_bits = (scores > thr).astype(int)

    metrics = {}
    if labels is not None:
        ev = event_f1(pred_bits, labels, iou_thresh=event_iou)
        auc_pr = pr_auc(scores, labels)
        metrics = dict(ev, auc_pr=auc_pr)

    out_dir = out_dir or os.path.join("outputs", "infer_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "scores.npy"), scores)
    with open(os.path.join(out_dir, "threshold.txt"), "w") as f:
        f.write(str(float(thr)))
    # 保存预测 csv
    out_csv = os.path.join(out_dir, "pred.csv")
    out_df = {"score": scores, "pred": pred_bits}
    if labels is not None:
        out_df["label"] = labels
    import pandas as pd
    pd.DataFrame(out_df).to_csv(out_csv, index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"threshold": float(thr), **metrics}, f, ensure_ascii=False, indent=2)

    return {"out_dir": out_dir, "threshold": float(thr), **metrics}
