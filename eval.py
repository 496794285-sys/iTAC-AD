#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAC-AD 评测脚本
支持POT阈值和事件级F1评测
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# 添加项目根路径
ROOT = Path("/Users/waba/PythonProject/Transformer Project/iTAC-AD").resolve()
sys.path.insert(0, str(ROOT))

from itac_ad.models.itac_ad import ITAC_AD
from itac_ad.evaluation import evaluate_model, save_evaluation_results

def load_model(checkpoint_path: Path, device: torch.device) -> ITAC_AD:
    """加载训练好的模型"""
    # 创建模型（需要知道特征数）
    feats = 7  # synthetic数据有7个特征
    model = ITAC_AD(feats=feats).to(device)
    
    # 触发模型构建（lazy_build需要前向传播）
    dummy_input = torch.randn(1, 96, 7).to(device)  # [B,T,D]
    with torch.no_grad():
        _ = model(dummy_input, phase=2, aac_w=0.0)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def create_test_dataloader(batch_size: int = 8, num_workers: int = 0):
    """创建测试数据加载器"""
    from vendor.tranad.main import TinySineDataset, make_loader
    
    # 创建测试数据集
    test_ds = TinySineDataset(N=32, T=96, D=7, noise=0.05)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)
    
    return test_loader

def main():
    parser = argparse.ArgumentParser(description="iTAC-AD 模型评测")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="模型checkpoint路径")
    parser.add_argument("--output_dir", "-o", type=str, default="./eval_output",
                       help="评测结果输出目录")
    parser.add_argument("--pot_q", type=float, default=0.98,
                       help="POT阈值分位数")
    parser.add_argument("--pot_level", type=float, default=0.99,
                       help="VaR水平")
    parser.add_argument("--event_iou", type=float, default=0.1,
                       help="事件IoU阈值")
    parser.add_argument("--score_reduction", type=str, default="mean",
                       choices=["mean", "median", "p95"],
                       help="分数聚合方式")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备 (cpu/cuda/mps/auto)")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 检查checkpoint文件
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: checkpoint文件不存在: {checkpoint_path}")
        sys.exit(1)
    
    # 加载模型
    print("加载模型...")
    model = load_model(checkpoint_path, device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建数据加载器
    print("创建测试数据...")
    test_loader = create_test_dataloader(batch_size=args.batch_size)
    
    # 运行评测
    print("开始评测...")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        pot_q=args.pot_q,
        pot_level=args.pot_level,
        event_iou=args.event_iou,
        score_reduction=args.score_reduction,
        true_labels=None  # synthetic数据没有真实标签
    )
    
    # 打印结果
    print("\n=== 评测结果 ===")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # 保存结果
    output_dir = Path(args.output_dir)
    save_evaluation_results(results, output_dir)
    print(f"\n结果已保存到: {output_dir}")
    
    print("\n✅ 评测完成！")

if __name__ == "__main__":
    main()
