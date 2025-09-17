#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAC调度器测试脚本
验证自适应对抗权重调度器的功能
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目根路径
ROOT = Path("/Users/waba/PythonProject/Transformer Project/iTAC-AD").resolve()
sys.path.insert(0, str(ROOT))

from itac_ad.components.aac_scheduler import AACScheduler

def test_aac_scheduler():
    """测试AAC调度器的功能"""
    print("=== AAC调度器测试开始 ===")
    
    # 创建AAC调度器
    aac = AACScheduler(
        tau=0.9,
        alpha=1.0,
        beta=0.5,
        window_size=32,  # 更小的窗口用于测试
        w_min=0.0,
        w_max=1.0,
        ema_decay=0.98,
        bins=16
    )
    
    print(f"AAC参数: tau={aac.tau}, alpha={aac.alpha}, beta={aac.beta}")
    print(f"窗口大小: {aac.window_size}, 预热阈值: {max(64, aac.window_size//4)}")
    
    # 模拟训练过程
    print("\n--- 模拟训练过程 ---")
    
    # 生成模拟残差数据
    np.random.seed(42)
    
    # 正常模式：低残差
    normal_residuals = np.random.normal(0.1, 0.05, 20)
    
    # 异常模式：高残差
    anomaly_residuals = np.random.normal(0.8, 0.2, 10)
    
    # 混合数据
    all_residuals = np.concatenate([normal_residuals, anomaly_residuals, normal_residuals])
    
    print("残差序列:", all_residuals[:10], "...")
    
    weights = []
    for i, residual_val in enumerate(all_residuals):
        # 创建tensor
        residual = torch.tensor([residual_val], dtype=torch.float32)
        
        # 更新AAC
        w = aac.step(residual)
        weights.append(w)
        
        stats = aac.stats()
        
        if i % 5 == 0 or i < 10:
            print(f"Step {i:2d}: residual={residual_val:.3f}, w={w:.3f}, "
                  f"q={stats['q']:.3f}, drift={stats['drift']:.3f}, "
                  f"buf_size={stats['buf_size']}")
    
    print(f"\n权重序列: {[f'{w:.3f}' for w in weights]}")
    
    # 验证AAC行为
    print("\n--- 验证AAC行为 ---")
    
    # 检查预热期
    warmup_threshold = max(64, aac.window_size//4)
    warmup_weights = weights[:warmup_threshold]
    warmup_all_zero = all(w == 0.0 for w in warmup_weights)
    print(f"预热期权重全为0: {warmup_all_zero}")
    
    # 检查异常检测
    if len(weights) > warmup_threshold:
        active_weights = weights[warmup_threshold:]
        print(f"活跃期权重范围: {min(active_weights):.3f} - {max(active_weights):.3f}")
        
        # 检查权重是否在异常段上升
        anomaly_start = 20
        anomaly_end = 30
        normal_start = 30
        
        if len(active_weights) > anomaly_end - warmup_threshold:
            anomaly_weights = active_weights[anomaly_start-warmup_threshold:anomaly_end-warmup_threshold]
            normal_weights = active_weights[normal_start-warmup_threshold:normal_start-warmup_threshold+5]
            
            avg_anomaly_w = np.mean(anomaly_weights)
            avg_normal_w = np.mean(normal_weights)
            
            print(f"异常段平均权重: {avg_anomaly_w:.3f}")
            print(f"正常段平均权重: {avg_normal_w:.3f}")
            print(f"异常段权重更高: {avg_anomaly_w > avg_normal_w}")
    
    print("\n✅ AAC调度器测试完成")
    return True

if __name__ == "__main__":
    success = test_aac_scheduler()
    if success:
        print("\n🎉 AAC调度器测试通过！")
        sys.exit(0)
    else:
        print("\n❌ AAC调度器测试失败！")
        sys.exit(1)
