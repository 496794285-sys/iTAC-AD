#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速体检：阈值与阳性比例调试工具
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from itacad.thresholding import fit_threshold

def debug_threshold(run_dir: str):
    """调试阈值和阳性比例"""
    run_path = Path(run_dir)
    
    print(f"=== 调试运行目录: {run_path} ===")
    
    # 1. 检查文件存在性
    files_to_check = [
        "threshold.json",
        "train_scores.npy", 
        "train_stats.json",
        "scores.npy",
        "labels.npy"
    ]
    
    print("\n1. 文件检查:")
    for fname in files_to_check:
        fpath = run_path / fname
        exists = fpath.exists()
        print(f"  {fname}: {'✓' if exists else '✗'}")
        if exists and fname.endswith('.npy'):
            try:
                data = np.load(fpath)
                print(f"    形状: {data.shape}, 类型: {data.dtype}")
            except Exception as e:
                print(f"    加载失败: {e}")
    
    # 2. 阈值信息
    print("\n2. 阈值信息:")
    thr_json = run_path / "threshold.json"
    if thr_json.exists():
        with open(thr_json, 'r') as f:
            thr_data = json.load(f)
        print(f"  训练阈值: {thr_data.get('threshold', 'N/A'):.4f}")
        print(f"  POT参数: q={thr_data.get('q', 'N/A')}, level={thr_data.get('level', 'N/A')}")
        print(f"  聚合方法: {thr_data.get('agg', 'N/A')}")
        print(f"  TopK比例: {thr_data.get('topk_ratio', 'N/A')}")
        
        if 'train_scores_stats' in thr_data:
            stats = thr_data['train_scores_stats']
            print(f"  训练分数统计: min={stats.get('min', 0):.4f}, max={stats.get('max', 0):.4f}, mean={stats.get('mean', 0):.4f}")
    else:
        print("  未找到threshold.json")
    
    # 3. 训练分数分析
    print("\n3. 训练分数分析:")
    train_scores_path = run_path / "train_scores.npy"
    if train_scores_path.exists():
        train_scores = np.load(train_scores_path)
        print(f"  训练分数: min={train_scores.min():.4f}, max={train_scores.max():.4f}")
        print(f"  训练分数: mean={train_scores.mean():.4f}, std={train_scores.std():.4f}")
        
        # 分位数分析
        quantiles = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
        print("  分位数:")
        for q in quantiles:
            val = np.quantile(train_scores, q)
            print(f"    {q*100:4.1f}%: {val:.4f}")
    else:
        print("  未找到train_scores.npy")
    
    # 4. 测试分数分析
    print("\n4. 测试分数分析:")
    test_scores_path = run_path / "scores.npy"
    if test_scores_path.exists():
        test_scores = np.load(test_scores_path)
        print(f"  测试分数: min={test_scores.min():.4f}, max={test_scores.max():.4f}")
        print(f"  测试分数: mean={test_scores.mean():.4f}, std={test_scores.std():.4f}")
        
        # 如果有阈值，计算阳性比例
        if thr_json.exists():
            with open(thr_json, 'r') as f:
                thr_data = json.load(f)
            threshold = thr_data.get('threshold', 0)
            positives = (test_scores > threshold).sum()
            positives_rate = positives / len(test_scores) * 100
            print(f"  阳性比例: {positives}/{len(test_scores)} ({positives_rate:.2f}%)")
            
            # 建议
            if positives_rate > 50:
                print("  ⚠️  阳性比例过高，建议降低POT_LVL或提高POT_Q")
            elif positives_rate < 0.1:
                print("  ⚠️  阳性比例过低，建议提高POT_LVL或降低POT_Q")
            else:
                print("  ✓ 阳性比例在合理范围内")
    else:
        print("  未找到scores.npy")
    
    # 5. 标签分析
    print("\n5. 标签分析:")
    labels_path = run_path / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path)
        anomaly_count = labels.sum()
        anomaly_rate = anomaly_count / len(labels) * 100
        print(f"  异常标签: {anomaly_count}/{len(labels)} ({anomaly_rate:.2f}%)")
    else:
        print("  未找到labels.npy")

def main():
    parser = argparse.ArgumentParser(description="调试阈值和阳性比例")
    parser.add_argument("--run", required=True, help="运行目录路径")
    args = parser.parse_args()
    
    debug_threshold(args.run)

if __name__ == "__main__":
    main()