# test_itac_ad_integration.py
import sys
import os
sys.path.insert(0, '/Users/waba/PythonProject/Transformer Project/iTAC-AD')

import torch
import torch.nn as nn
from itac_ad.models.itac_ad import ITAC_AD
from itac_ad.components.aac_scheduler import AACScheduler

def test_itac_ad_basic():
    """测试 iTAC-AD 模型的基本功能"""
    print("测试 iTAC-AD 模型基本功能...")
    
    # 创建模型
    model = ITAC_AD(feats=7, d_model=128, n_heads=8, e_layers=2, dropout=0.1)
    print(f"模型名称: {model.name}")
    print(f"学习率: {model.lr}")
    print(f"批次大小: {model.batch}")
    print(f"特征数: {model.n_feats}")
    print(f"窗口大小: {model.n_window}")
    
    # 创建 AAC 调度器
    aac = AACScheduler(window_size=256, quantile_p=0.9)
    
    # 测试前向传播
    batch_size = 4
    seq_len = 10
    feats = 7
    
    # 创建测试数据
    x = torch.randn(batch_size, seq_len, feats)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    o1, o2, aux = model(x)
    print(f"Phase-1 输出形状: {o1.shape}")
    print(f"Phase-2 输出形状: {o2.shape}")
    print(f"辅助信息键: {list(aux.keys())}")
    
    # 测试 AAC 调度器
    residual = torch.abs(o1 - x)
    w_t = aac.step(residual)
    print(f"AAC 权重: {w_t:.4f}")
    
    # 测试损失计算
    loss_rec = nn.MSELoss()(o1, x)
    loss_adv = -nn.MSELoss()(o2, x)
    total_loss = loss_rec + w_t * loss_adv
    
    print(f"重构损失: {loss_rec.item():.4f}")
    print(f"对抗损失: {loss_adv.item():.4f}")
    print(f"总损失: {total_loss.item():.4f}")
    
    print("✅ iTAC-AD 基本功能测试通过！")

def test_tranad_compatibility():
    """测试与 TranAD 的兼容性"""
    print("\n测试与 TranAD 的兼容性...")
    
    # 模拟 TranAD 的数据格式
    batch_size = 2
    seq_len = 10
    feats = 7
    
    # TranAD 格式: [seq_len, batch_size, feats]
    tranad_data = torch.randn(seq_len, batch_size, feats)
    print(f"TranAD 格式数据形状: {tranad_data.shape}")
    
    # 转换为 iTAC-AD 格式: [batch_size, seq_len, feats]
    itac_data = tranad_data.permute(1, 0, 2)
    print(f"iTAC-AD 格式数据形状: {itac_data.shape}")
    
    # 创建模型
    model = ITAC_AD(feats=feats, d_model=128, n_heads=8, e_layers=2, dropout=0.1)
    
    # 前向传播
    o1, o2, aux = model(itac_data)
    
    # 模拟 TranAD 的目标格式
    elem = tranad_data[-1, :, :].view(1, batch_size, feats)  # [1, batch_size, feats]
    print(f"目标形状: {elem.shape}")
    
    # 计算损失 - 修正形状匹配
    # elem.squeeze(0) 形状是 [batch_size, feats]，需要扩展为 [batch_size, seq_len, feats]
    target = elem.squeeze(0).unsqueeze(1).expand(-1, seq_len, -1)
    loss_rec = nn.MSELoss()(o1, target)
    loss_adv = -nn.MSELoss()(o2, target)
    
    print(f"重构损失: {loss_rec.item():.4f}")
    print(f"对抗损失: {loss_adv.item():.4f}")
    
    print("✅ TranAD 兼容性测试通过！")

if __name__ == "__main__":
    test_itac_ad_basic()
    test_tranad_compatibility()
    print("\n🎉 所有测试通过！iTAC-AD 已成功集成到 TranAD 框架中。")
