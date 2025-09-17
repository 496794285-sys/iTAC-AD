# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

try:
    from scipy.stats import genpareto
except Exception:  # 允许没有 scipy 时的兜底
    genpareto = None

@dataclass
class Event:
    start: int
    end: int

def _to_events(bits: np.ndarray) -> List[Event]:
    """把0/1序列合并成事件片段（闭区间: start..end）。"""
    bits = bits.astype(bool)
    events = []
    i, n = 0, len(bits)
    while i < n:
        if bits[i]:
            j = i
            while j + 1 < n and bits[j + 1]:
                j += 1
            events.append(Event(i, j))
            i = j + 1
        else:
            i += 1
    return events

def _iou(e1: Event, e2: Event) -> float:
    inter = max(0, min(e1.end, e2.end) - max(e1.start, e2.start) + 1)
    union = (e1.end - e1.start + 1) + (e2.end - e2.start + 1) - inter
    return inter / union if union > 0 else 0.0

def pot_threshold(scores: np.ndarray, q: float = 0.98, level: float = 0.99) -> float:
    """POT: 更稳的阈值计算，失败回退到高分位"""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return float(np.quantile(s, level))

    # 基本分位与尾部
    q = float(np.clip(q, 0.80, 0.995))
    level = float(np.clip(level, max(q + 1e-3, 0.90), 0.9999))
    u = np.quantile(s, q)
    tail = s[s > u] - u

    # 回退条件：尾部太少 / 方差过小 / GPD 不稳定
    if tail.size < 30 or tail.std() < 1e-8 or genpareto is None:
        return float(np.quantile(s, level))

    try:
        c, loc, scale = genpareto.fit(tail, floc=0.0)
        if not np.isfinite(scale) or scale <= 1e-12:
            return float(np.quantile(s, level))
        p_tail = max(1e-9, 1.0 - q)
        ppf_arg = float(np.clip((level - q) / p_tail, 1e-6, 1.0 - 1e-9))
        var = u + genpareto.ppf(ppf_arg, c, 0.0, scale)
        if not np.isfinite(var):
            return float(np.quantile(s, level))
        # 阈值至少不低于 u，也不高于 max(s)
        return float(np.clip(var, u, s.max()))
    except Exception:
        return float(np.quantile(s, level))

def binarize_by_threshold(scores: np.ndarray, thr: float) -> np.ndarray:
    return (np.asarray(scores) > float(thr)).astype(int)

def event_f1(
    pred_bits: np.ndarray,
    true_bits: np.ndarray,
    iou_thresh: float = 0.1
) -> Dict[str, float]:
    """事件级匹配：IoU≥阈值视为命中。"""
    pred_events = _to_events(pred_bits)
    true_events = _to_events(true_bits)
    if not pred_events and not true_events:
        return dict(precision=1.0, recall=1.0, f1=1.0, tp=0, fp=0, fn=0)
    used_true = set()
    tp = 0
    for pe in pred_events:
        hit = -1
        best = 0.0
        for i, te in enumerate(true_events):
            if i in used_true: 
                continue
            iou = _iou(pe, te)
            if iou >= iou_thresh and iou > best:
                best, hit = iou, i
        if hit >= 0:
            tp += 1
            used_true.add(hit)
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)

def point_f1(pred_bits, true_bits):
    """点级F1指标计算"""
    y, p = true_bits.astype(int), pred_bits.astype(int)
    tp = int((p & (y==1)).sum())
    fp = int((p & (y==0)).sum())
    fn = int(((1-p) & (y==1)).sum())
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall    = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return dict(point_precision=precision, point_recall=recall, point_f1=f1, tp=tp, fp=fp, fn=fn)

def pr_auc(scores: np.ndarray, labels: np.ndarray, num_thresh: int = 512) -> float:
    """简单PR AUC：阈值扫一遍（从低到高），点积近似积分。"""
    s = np.asarray(scores).astype(float)
    y = np.asarray(labels).astype(int)
    if y.sum() == 0:
        return 0.0
    thrs = np.linspace(s.min(), s.max(), num=num_thresh)
    P, R = [], []
    for t in thrs:
        pred = (s > t).astype(int)
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        fn = ((1 - pred) & (y == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        P.append(precision); R.append(recall)
    # R递增排序再梯形积分
    P = np.array(P); R = np.array(R)
    order = np.argsort(R)
    P, R = P[order], R[order]
    auc = np.trapz(P, R)
    return float(max(0.0, min(1.0, auc)))
