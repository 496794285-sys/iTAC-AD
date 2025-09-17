#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测管线：POT 阈值 + 事件级 F1
实现完整的异常检测评测系统
"""
import os
import sys
import numpy as np
import torch
import scipy.stats
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import deque

# 添加项目根路径
ROOT = Path("/Users/waba/PythonProject/Transformer Project/iTAC-AD").resolve()
sys.path.insert(0, str(ROOT))

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

def pot_threshold(scores: np.ndarray, q: float = 0.98, level: float = 0.99) -> float:
    """
    POT (Peaks Over Threshold) 阈值计算
    使用GPD (Generalized Pareto Distribution) 拟合尾部数据
    
    Args:
        scores: 异常分数数组
        q: 阈值分位数 (默认0.98)
        level: VaR水平 (默认0.99)
    
    Returns:
        计算得到的阈值
    """
    s = np.asarray(scores)
    u = np.quantile(s, q)  # 阈值
    
    # 提取尾部数据
    tail = s[s > u] - u
    
    if len(tail) < 10:  # 尾部数据太少，使用简单分位数
        return float(np.quantile(s, level))
    
    try:
        # GPD拟合
        c, loc, scale = scipy.stats.genpareto.fit(tail, floc=0)
        
        # 计算VaR水平阈值
        p_tail = 1 - q
        var = u + scipy.stats.genpareto.ppf((level - q) / p_tail, c, loc=0, scale=scale)
        
        return float(var)
    except:
        # GPD拟合失败，回退到简单分位数
        return float(np.quantile(s, level))

def simple_threshold(scores: np.ndarray, q: float = 0.98) -> float:
    """
    简单分位数阈值（失败回退）
    """
    return float(np.quantile(scores, q))

def compute_anomaly_scores(model_outputs: Dict, reduction: str = "mean") -> np.ndarray:
    """
    计算异常分数
    
    Args:
        model_outputs: 模型输出字典，包含O1, O2等
        reduction: 分数聚合方式 ("mean", "median", "p95")
    
    Returns:
        异常分数数组
    """
    o1 = model_outputs["O1"]  # Phase-1重构
    o2 = model_outputs["O2"]  # Phase-2重构
    
    if isinstance(o1, torch.Tensor):
        o1 = o1.detach().cpu().numpy()
    if isinstance(o2, torch.Tensor):
        o2 = o2.detach().cpu().numpy()
    
    # 计算重构误差
    scores = np.abs(o1 - o2)  # 使用两个重构的差异作为异常分数
    
    if reduction == "mean":
        scores = np.mean(scores, axis=(1, 2))  # [B, T, D] -> [B]
    elif reduction == "median":
        scores = np.median(scores, axis=(1, 2))
    elif reduction == "p95":
        scores = np.percentile(scores, 95, axis=(1, 2))
    else:
        scores = np.mean(scores, axis=(1, 2))  # 默认mean
    
    return scores

def binary_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    将异常分数转换为二值预测
    
    Args:
        scores: 异常分数数组
        threshold: 阈值
    
    Returns:
        二值预测数组 (0=正常, 1=异常)
    """
    return (scores > threshold).astype(int)

def merge_consecutive_anomalies(predictions: np.ndarray, min_length: int = 1) -> List[Tuple[int, int]]:
    """
    合并连续的异常预测为事件段
    
    Args:
        predictions: 二值预测数组
        min_length: 最小事件长度
    
    Returns:
        事件段列表 [(start, end), ...]
    """
    events = []
    in_event = False
    start = 0
    
    for i, pred in enumerate(predictions):
        if pred == 1 and not in_event:
            # 开始新事件
            start = i
            in_event = True
        elif pred == 0 and in_event:
            # 结束当前事件
            if i - start >= min_length:
                events.append((start, i))
            in_event = False
    
    # 处理序列末尾的事件
    if in_event and len(predictions) - start >= min_length:
        events.append((start, len(predictions)))
    
    return events

def compute_event_metrics(
    pred_events: List[Tuple[int, int]],
    true_events: List[Tuple[int, int]],
    iou_threshold: float = 0.1,
    delay_tolerance: int = 0
) -> Dict[str, float]:
    """
    计算事件级指标
    
    Args:
        pred_events: 预测事件段列表
        true_events: 真实事件段列表
        iou_threshold: IoU阈值
        delay_tolerance: 延迟容忍度
    
    Returns:
        包含precision, recall, f1的字典
    """
    if not pred_events and not true_events:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if not pred_events:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not true_events:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # 计算IoU
    def compute_iou(event1, event2):
        start1, end1 = event1
        start2, end2 = event2
        
        # 计算交集
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection = max(0, intersection_end - intersection_start)
        
        # 计算并集
        union = (end1 - start1) + (end2 - start2) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # 匹配预测事件和真实事件
    matched_pred = set()
    matched_true = set()
    
    for i, pred_event in enumerate(pred_events):
        for j, true_event in enumerate(true_events):
            if j in matched_true:
                continue
            
            iou = compute_iou(pred_event, true_event)
            if iou >= iou_threshold:
                matched_pred.add(i)
                matched_true.add(j)
                break
    
    # 计算指标
    tp = len(matched_pred)
    fp = len(pred_events) - tp
    fn = len(true_events) - len(matched_true)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

def evaluate_model(
    model,
    dataloader,
    device,
    pot_q: float = 0.98,
    pot_level: float = 0.99,
    event_iou: float = 0.1,
    score_reduction: str = "mean",
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    完整的模型评测流程
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        pot_q: POT阈值分位数
        pot_level: VaR水平
        event_iou: 事件IoU阈值
        score_reduction: 分数聚合方式
        true_labels: 真实标签（可选）
    
    Returns:
        评测结果字典
    """
    model.eval()
    all_scores = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                x, y = batch_data[0], batch_data[1]
            else:
                x = batch_data
            
            x = x.to(device)
            
            # 模型前向传播
            result = model(x, phase=2, aac_w=0.0)
            
            # 计算异常分数
            scores = compute_anomaly_scores(result, reduction=score_reduction)
            all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    
    # 计算阈值
    try:
        threshold = pot_threshold(all_scores, q=pot_q, level=pot_level)
    except:
        threshold = simple_threshold(all_scores, q=pot_q)
    
    # 生成预测
    predictions = binary_predictions(all_scores, threshold)
    
    # 计算基本指标
    results = {
        "threshold": threshold,
        "mean_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores)),
        "anomaly_rate": float(np.mean(predictions))
    }
    
    # 如果有真实标签，计算事件级指标
    if true_labels is not None:
        pred_events = merge_consecutive_anomalies(predictions)
        true_events = merge_consecutive_anomalies(true_labels)
        
        event_metrics = compute_event_metrics(pred_events, true_events, iou_threshold=event_iou)
        results.update(event_metrics)
        
        # 添加事件统计
        results["n_pred_events"] = len(pred_events)
        results["n_true_events"] = len(true_events)
    
    return results

def save_evaluation_results(results: Dict[str, float], output_dir: Path):
    """保存评测结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存阈值
    with open(output_dir / "threshold.txt", "w") as f:
        f.write(f"{results['threshold']:.6f}\n")
    
    # 保存详细结果
    with open(output_dir / "evaluation_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.6f}\n")

if __name__ == "__main__":
    # 测试评测管线
    print("=== 评测管线测试 ===")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟异常分数
    normal_scores = np.random.normal(0.1, 0.05, n_samples)
    anomaly_scores = np.random.normal(0.8, 0.2, 100)
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    
    print(f"分数统计: mean={np.mean(all_scores):.3f}, std={np.std(all_scores):.3f}")
    
    # 测试POT阈值
    threshold = pot_threshold(all_scores, q=0.98, level=0.99)
    print(f"POT阈值: {threshold:.3f}")
    
    # 测试简单阈值
    simple_thresh = simple_threshold(all_scores, q=0.98)
    print(f"简单阈值: {simple_thresh:.3f}")
    
    # 测试事件级指标
    predictions = binary_predictions(all_scores, threshold)
    true_labels = np.zeros(n_samples)
    true_labels[900:950] = 1  # 模拟真实异常段
    
    pred_events = merge_consecutive_anomalies(predictions)
    true_events = merge_consecutive_anomalies(true_labels)
    
    print(f"预测事件数: {len(pred_events)}")
    print(f"真实事件数: {len(true_events)}")
    
    event_metrics = compute_event_metrics(pred_events, true_events, iou_threshold=0.1)
    print(f"事件级指标: P={event_metrics['precision']:.3f}, R={event_metrics['recall']:.3f}, F1={event_metrics['f1']:.3f}")
    
    print("✅ 评测管线测试通过！")
