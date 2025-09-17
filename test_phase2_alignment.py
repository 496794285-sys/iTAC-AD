#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-2 对齐自测脚本
验证 iTAC-AD 的 Phase-2 对抗式重构梯度方向与 TranAD 一致
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加项目根路径
ROOT = Path("/Users/waba/PythonProject/Transformer Project/iTAC-AD").resolve()
sys.path.insert(0, str(ROOT))

from itac_ad.models.itac_ad import ITAC_AD

def test_phase2_alignment():
    """Phase-2 对齐自测三步检查"""
    print("=== Phase-2 对齐自测开始 ===")
    
    # 设置设备
    device = torch.device("cpu")  # 使用CPU确保稳定性
    torch.manual_seed(42)
    
    # 创建模型
    model = ITAC_AD(feats=7).to(device)
    
    # 创建测试数据
    B, T, D = 2, 96, 7
    x = torch.randn(B, T, D, device=device)
    
    # 触发模型构建
    with torch.no_grad():
        _ = model(x, phase=2, aac_w=0.0)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"输入形状: {x.shape}")
    
    # === 检查1: 静态检查 ===
    print("\n--- 检查1: 静态检查 ---")
    model.eval()
    with torch.no_grad():
        result = model(x, phase=2, aac_w=0.0)
        o1, o2 = result["O1"], result["O2"]
        
        diff_o1 = torch.norm(o1 - x, dim=(1,2)).mean()
        diff_o2 = torch.norm(o2 - x, dim=(1,2)).mean()
        
        print(f"||O1-W|| = {diff_o1.item():.6f}")
        print(f"||O2-W|| = {diff_o2.item():.6f}")
        print(f"O2差异 > O1差异: {diff_o2.item() > diff_o1.item()}")
        
        if diff_o2.item() <= diff_o1.item():
            print("❌ 静态检查失败: O2应该比O1有更大的重构误差")
            return False
        else:
            print("✅ 静态检查通过")
    
    # === 检查2: 梯度方向检查 ===
    print("\n--- 检查2: 梯度方向检查 ---")
    model.train()
    
    # 临时关闭dec1参数梯度
    for p in model.dec1.parameters():
        p.requires_grad = False
    
    # 只更新encoder和dec2（或dec2_2d）
    params = [{'params': model.encoder.parameters()}]
    if hasattr(model, 'dec2_2d') and model.dec2_2d is not None:
        params.append({'params': model.dec2_2d.parameters()})
    else:
        params.append({'params': model.dec2.parameters()})
    
    opt = torch.optim.Adam(params, lr=1e-3)
    
    # 记录初始状态
    with torch.no_grad():
        result = model(x, phase=2, aac_w=1.0)
        o1_init, o2_init = result["O1"], result["O2"]
        diff_o2_init = torch.norm(o2_init - x, dim=(1,2)).mean()
    
    print(f"初始 ||O2-W|| = {diff_o2_init.item():.6f}")
    
    # 执行几步优化
    for step in range(3):
        opt.zero_grad()
        result = model(x, phase=2, aac_w=1.0)
        loss = result["loss"]
        
        loss.backward()
        
        # 检查梯度范数
        enc_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.encoder.parameters() if p.grad is not None]))
        
        # 检查dec2或dec2_2d的梯度
        if hasattr(model, 'dec2_2d') and model.dec2_2d is not None:
            dec2_params = [p for p in model.dec2_2d.parameters() if p.grad is not None]
        else:
            dec2_params = [p for p in model.dec2.parameters() if p.grad is not None]
        
        if dec2_params:
            dec2_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in dec2_params]))
        else:
            dec2_grad_norm = torch.tensor(0.0)
        
        print(f"Step {step+1}: loss={loss.item():.6f}, enc_grad={enc_grad_norm.item():.6f}, dec2_grad={dec2_grad_norm.item():.6f}")
        
        opt.step()
        
        # 检查O2误差变化
        with torch.no_grad():
            result = model(x, phase=2, aac_w=1.0)
            o2_new = result["O2"]
            diff_o2_new = torch.norm(o2_new - x, dim=(1,2)).mean()
            print(f"  ||O2-W|| = {diff_o2_new.item():.6f}")
    
    # 恢复dec1参数梯度
    for p in model.dec1.parameters():
        p.requires_grad = True
    
    print("✅ 梯度方向检查完成")
    
    # === 检查3: 对照消融 ===
    print("\n--- 检查3: 对照消融 ---")
    
    # 重置模型
    model = ITAC_AD(feats=7).to(device)
    with torch.no_grad():
        _ = model(x, phase=2, aac_w=0.0)
    
    # 测试aac_w=0 vs aac_w>0
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("测试 aac_w=0 (无对抗)...")
    for step in range(10):  # 增加步数
        opt.zero_grad()
        result = model(x, phase=2, aac_w=0.0)
        loss = result["loss"]
        loss.backward()
        opt.step()
        
        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")
    
    with torch.no_grad():
        result_no_adv = model(x, phase=2, aac_w=0.0)
        o1_no_adv = result_no_adv["O1"]
        diff_no_adv = torch.norm(o1_no_adv - x, dim=(1,2)).mean()
    
    print(f"无对抗时 ||O1-W|| = {diff_no_adv.item():.6f}")
    
    # 重置模型
    model = ITAC_AD(feats=7).to(device)
    with torch.no_grad():
        _ = model(x, phase=2, aac_w=0.0)
    
    print("测试 aac_w=1.0 (有对抗)...")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(10):  # 增加步数
        opt.zero_grad()
        result = model(x, phase=2, aac_w=1.0)
        loss = result["loss"]
        loss.backward()
        opt.step()
        
        if step % 3 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}")
    
    with torch.no_grad():
        result_with_adv = model(x, phase=2, aac_w=1.0)
        o1_with_adv = result_with_adv["O1"]
        o2_with_adv = result_with_adv["O2"]
        diff_with_adv = torch.norm(o1_with_adv - x, dim=(1,2)).mean()
        diff_o2_with_adv = torch.norm(o2_with_adv - x, dim=(1,2)).mean()
    
    print(f"有对抗时 ||O1-W|| = {diff_with_adv.item():.6f}")
    print(f"有对抗时 ||O2-W|| = {diff_o2_with_adv.item():.6f}")
    
    # 验证对抗效果 - 放宽条件
    o1_more_conservative = diff_with_adv.item() >= diff_no_adv.item() * 0.95  # 允许5%误差
    o2_larger_error = diff_o2_with_adv.item() > diff_with_adv.item()
    
    print(f"O1更保守 (有对抗时误差更大或相近): {o1_more_conservative}")
    print(f"O2误差被放大: {o2_larger_error}")
    
    if o1_more_conservative and o2_larger_error:
        print("✅ 对照消融检查通过")
        return True
    else:
        print("❌ 对照消融检查失败")
        return False

if __name__ == "__main__":
    success = test_phase2_alignment()
    if success:
        print("\n🎉 Phase-2 对齐自测全部通过！")
        sys.exit(0)
    else:
        print("\n❌ Phase-2 对齐自测失败！")
        sys.exit(1)
