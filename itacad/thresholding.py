# -*- coding: utf-8 -*-
"""
itacad/thresholding.py
一个快速、稳健、零-SciPy 依赖的阈值拟合模块。

核心接口：
  - fit_threshold(scores, q=0.98, level=0.99, method="auto", ...)
      -> thr, meta
  - save_threshold(run_dir, thr, meta)
  - load_threshold(run_dir) -> (thr or None, meta)

设计要点：
  * 只用 NumPy；默认用 Pickands 闭式估计（GPD），毫秒级出结果。
  * 对尾部：随机降采样 + 极端值裁剪，避免尖峰干扰。
  * 多重回退：Pickands 估计不稳 → 直接高分位（quantile）。
  * 所有参数均可显式传入，或用环境变量覆盖（不强制）。
"""

from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
import numpy as np

# -------- 默认参数（可用环境变量覆盖） -----------------------------------------

_DEF_Q        = float(os.environ.get("POT_Q",        "0.98"))   # 选阈分位
_DEF_LEVEL    = float(os.environ.get("POT_LVL",      "0.99"))   # 目标报警分位
_DEF_MAX_TAIL = int  (os.environ.get("POT_MAX_TAIL", "5000"))   # 尾部采样上限
_DEF_TOP_CLIP = float(os.environ.get("POT_TOP_CLIP", "0.999"))  # 尾部极值裁剪分位
_DEF_SEED     = int  (os.environ.get("POT_SEED",     "0"))      # 采样种子
_DEF_METHOD   =       os.environ.get("POT_MODE",     "auto").lower()  # auto|pickands|quantile
_DEF_K        = int  (os.environ.get("POT_K",        "2000"))   # Pickands 用的 k（近似"阶次"）

# -------- 数据类：返回的元信息 -----------------------------------------------

@dataclass
class ThresholdMeta:
    method: str
    q: float
    level: float
    max_tail: int
    top_clip: float
    seed: int
    k: int
    u: float                  # 基准阈 u = quantile(scores, q)
    tail_size: int            # 用于拟合的尾部点数（采样/裁剪后）
    fallback: Optional[str]   # 若发生回退，这里写原因；否则 None

# -------- 工具函数 ------------------------------------------------------------

def _clean_scores(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    return s

def _quantile(s: np.ndarray, p: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if s.size == 0: return 0.0
    return float(np.quantile(s, p))

def _prepare_tail(s: np.ndarray, q: float, max_tail: int, top_clip: float, seed: int) -> Tuple[float, np.ndarray]:
    """返回 (u, tail_exceedances)，其中 tail 已采样+裁剪。"""
    q = float(np.clip(q, 0.80, 0.995))                # 给个合理区间
    u = _quantile(s, q)
    tail = s[s > u] - u
    m = tail.size
    if m == 0:
        return u, tail
    # 随机下采样（避免超长尾）
    if m > max_tail:
        rng = np.random.default_rng(seed)
        idx = rng.choice(m, size=max_tail, replace=False)
        tail = tail[idx]
        m = max_tail
    # 极端值裁掉 0.1% 尖峰，稳住闭式估计
    if m > 50 and 0.5 < top_clip < 1.0:
        hi = np.quantile(tail, top_clip)
        tail = np.clip(tail, 0.0, hi)
    return float(u), np.asarray(tail, dtype=np.float64)

# -------- Pickands 闭式估计（GPD），超快 --------------------------------------

def _pickands_quantile(s: np.ndarray, *, q: float, level: float, k: int,
                       max_tail: int, top_clip: float, seed: int) -> Tuple[Optional[float], ThresholdMeta]:
    """
    用 Pickands 闭式估计 GPD 形状 xi 与尺度 beta，并给出目标分位对应阈值。
    返回 (thr 或 None, meta)
    """
    s = _clean_scores(s)
    level = float(np.clip(level, max(q + 1e-3, 0.90), 0.9999))

    u, tail = _prepare_tail(s, q, max_tail, top_clip, seed)
    m = tail.size
    meta = ThresholdMeta(method="pickands", q=q, level=level,
                         max_tail=max_tail, top_clip=top_clip, seed=seed,
                         k=k, u=u, tail_size=m, fallback=None)

    # 尾部太短，Pickands 不稳：交给上层回退
    if m < 200:
        meta.fallback = "tail_too_short"
        return None, meta

    # 选择 k：默认 2000；若样本很少，取 m//20 ~ m//4 区间
    k = int(np.clip(k, 50, max(50, m // 4)))
    k = int(max(50, min(k, max(50, m // 10))))  # 介于 [50, m/10] 内
    # 取序统计量
    y = np.sort(tail)  # 升序
    # 需要 y[k-1], y[2k-1], y[4k-1] 可用
    if 4 * k - 1 >= m:
        k = m // 5
        if k < 50:
            meta.fallback = "k_too_large"
            return None, meta
    x1, x2, x4 = y[k - 1], y[2 * k - 1], y[4 * k - 1]
    if not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(x4)):
        meta.fallback = "nan_in_order_stats"
        return None, meta
    if (x2 - x1) <= 0 or (x4 - x2) <= 0:
        meta.fallback = "non_increasing_gaps"
        return None, meta

    xi = (1.0 / np.log(2.0)) * np.log((x4 - x2) / (x2 - x1))
    # 稳定化：限制在 [-0.2, 1.0]（工程上足够用）
    xi = float(np.clip(xi, -0.2, 1.0))
    beta = (2.0 ** xi - 1.0) * (x2 - x1)
    beta = float(max(beta, 1e-12))

    t = (level - q) / max(1e-12, 1.0 - q)  # 目标在尾部的分位比
    if abs(xi) < 1e-6:                     # xi→0 退化为指数尾
        thr = u + beta * np.log(1.0 / (1.0 - t))
    else:
        thr = u + beta / xi * ((1.0 - t) ** (-xi) - 1.0)

    # 最终裁剪到 [u, max(s)]，防止离谱
    thr = float(np.clip(thr, u, float(s.max()))) if s.size else float(thr)
    if not np.isfinite(thr):
        meta.fallback = "thr_not_finite"
        return None, meta
    return thr, meta

# -------- 对外主接口 ---------------------------------------------------------

def fit_threshold(
    scores: np.ndarray,
    q: float = _DEF_Q,
    level: float = _DEF_LEVEL,
    method: str = _DEF_METHOD,             # "auto" | "pickands" | "quantile"
    *,
    max_tail: int = _DEF_MAX_TAIL,
    top_clip: float = _DEF_TOP_CLIP,
    seed: int = _DEF_SEED,
    k: int = _DEF_K,
) -> Tuple[float, Dict]:
    """
    基于分数序列拟合报警阈值。

    参数
    ----
    scores : np.ndarray     分数序列（任意长度，包含 NaN/inf 会被清理）
    q      : float          基准阈 u 的分位（默认 0.98）
    level  : float          目标报警分位（默认 0.99）
    method : str            "auto"（默认）、"pickands"、或 "quantile"
    max_tail : int          尾部抽样上限（默认 5000）
    top_clip : float        尾部极值裁剪分位（默认 0.999）
    seed   : int            随机种子（用于尾部抽样）
    k      : int            Pickands 阶次参数（默认 2000）

    返回
    ----
    (thr, meta_dict)
      thr : float           拟合得到的报警阈值
      meta_dict : Dict      元信息（可直接写入 JSON）
    """
    s = _clean_scores(scores)
    # 极小样本：直接高分位
    if s.size < 10:
        thr = _quantile(s, float(level))
        meta = ThresholdMeta(method="quantile", q=q, level=level,
                             max_tail=max_tail, top_clip=top_clip, seed=seed,
                             k=k, u=_quantile(s, q), tail_size=0, fallback="too_few_points")
        return thr, asdict(meta)

    q = float(np.clip(q, 0.80, 0.995))
    level = float(np.clip(level, max(q + 1e-3, 0.90), 0.9999))

    # 选择方法
    m = method.lower()
    if m in ("auto", "pickands"):
        thr, meta = _pickands_quantile(
            s, q=q, level=level, k=k,
            max_tail=max_tail, top_clip=top_clip, seed=seed
        )
        if thr is not None:
            return thr, asdict(meta)

        # Pickands 不稳 → 回退到高分位
        thr = _quantile(s, level)
        meta.fallback = f"fallback_to_quantile:{meta.fallback or 'unknown'}"
        meta.method = "quantile"
        return thr, asdict(meta)

    # 纯高分位（最稳，阈值更保守）
    thr = _quantile(s, level)
    meta = ThresholdMeta(method="quantile", q=q, level=level,
                         max_tail=max_tail, top_clip=top_clip, seed=seed,
                         k=k, u=_quantile(s, q), tail_size=int((s > _quantile(s, q)).sum()),
                         fallback=None)
    return thr, asdict(meta)

# -------- 存取 JSON ----------------------------------------------------------

def save_threshold(run_dir: str, thr: float, meta: Dict):
    os.makedirs(run_dir, exist_ok=True)
    payload = {"threshold": float(thr), **meta}
    with open(os.path.join(run_dir, "threshold.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def load_threshold(run_dir: str) -> Tuple[Optional[float], Dict]:
    p = os.path.join(run_dir, "threshold.json")
    if not os.path.exists(p): return None, {}
    with open(p, "r") as f:
        d = json.load(f)
    return float(d.get("threshold")), d
