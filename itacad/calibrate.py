# -*- coding: utf-8 -*-
import os, json, numpy as np
from typing import Dict, Tuple
try:
    from scipy.stats import genpareto
except Exception:
    genpareto = None

def robust_pot(scores: np.ndarray, q: float=0.98, level: float=0.99) -> float:
    """
    稳健的POT（Peaks Over Threshold）阈值估计
    
    Args:
        scores: 分数数组
        q: POT阈值分位数（默认0.98）
        level: 目标异常检测水平（默认0.99）
    
    Returns:
        估计的阈值
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if s.size < 10: 
        return float(np.quantile(s, level))
    
    q = float(np.clip(q, 0.80, 0.995))
    level = float(np.clip(level, max(q+1e-3, 0.90), 0.9999))
    
    u = np.quantile(s, q)
    tail = s[s>u] - u
    
    if tail.size < 30 or tail.std()<1e-8 or genpareto is None:
        return float(np.quantile(s, level))
    
    try:
        c, loc, scale = genpareto.fit(tail, floc=0.0)
        if not np.isfinite(scale) or scale<=1e-12:
            return float(np.quantile(s, level))
        
        p_tail = max(1e-9, 1.0-q)
        arg = float(np.clip((level-q)/p_tail, 1e-6, 1-1e-9))
        thr = u + genpareto.ppf(arg, c, 0.0, scale)
        return float(np.clip(thr, u, s.max()))
    except Exception:
        return float(np.quantile(s, level))

def save_threshold(run_dir: str, thr: float, meta: Dict):
    """
    保存阈值到JSON文件
    
    Args:
        run_dir: 运行目录
        thr: 阈值
        meta: 元数据
    """
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(thr), **meta}, f, indent=2, ensure_ascii=False)

def load_threshold(run_dir: str) -> Tuple[float, Dict]:
    """
    从JSON文件加载阈值
    
    Args:
        run_dir: 运行目录
    
    Returns:
        (阈值, 元数据)
    """
    thr_json = os.path.join(run_dir, "threshold.json")
    if os.path.exists(thr_json):
        with open(thr_json, "r") as f:
            data = json.load(f)
            return float(data["threshold"]), data
    return None, {}
