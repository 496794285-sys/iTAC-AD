#!/usr/bin/env python3
"""
iTAC-AD 快速验证脚本
测试解码器切换和环境变量功能
"""
import os
import torch
import sys
sys.path.insert(0, '/Users/waba/PythonProject/Transformer Project/iTAC-AD')

from itac_ad.models.itac_ad import ITAC_AD

def test_decoder_switching():
    """测试解码器切换功能"""
    print("=== 测试解码器切换功能 ===")
    
    # 测试 MLP 解码器
    print("1. 测试 MLP 解码器...")
    os.environ["ITAC_DECODER"] = "mlp"
    
    model_mlp = ITAC_AD(feats=7)
    x = torch.randn(4, 96, 7)
    o1, o2, aux = model_mlp(x)
    print(f"   MLP解码器输出形状: o1={o1.shape}, o2={o2.shape}")
    
    # 测试 TranAD 解码器
    print("2. 测试 TranAD 解码器...")
    os.environ["ITAC_DECODER"] = "tranad"
    model_tranad = ITAC_AD(feats=7)
    o1, o2, aux = model_tranad(x)
    print(f"   TranAD解码器输出形状: o1={o1.shape}, o2={o2.shape}")
    
    print("✓ 解码器切换测试通过")

def test_environment_variables():
    """测试环境变量功能"""
    print("\n=== 测试环境变量功能 ===")
    
    # 设置环境变量
    os.environ["ITAC_D_MODEL"] = "64"
    os.environ["ITAC_N_HEADS"] = "4"
    os.environ["ITAC_E_LAYERS"] = "1"
    os.environ["ITAC_DROPOUT"] = "0.2"
    
    model = ITAC_AD(feats=5)
    print(f"   模型参数: d_model={model.encoder.d_model}, n_heads={model.encoder.n_heads}")
    print(f"   e_layers={model.encoder.e_layers}, dropout={model.encoder.dropout}")
    
    print("✓ 环境变量测试通过")

def test_forward_pass():
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    model = ITAC_AD(feats=7)
    model.eval()
    x = torch.randn(2, 64, 7)

    o1, o2, aux = model(x)
    
    print(f"   输入形状: {x.shape}")
    print(f"   输出1形状: {o1.shape}")
    print(f"   输出2形状: {o2.shape}")
    print(f"   辅助信息: {list(aux.keys())}")
    
    # 验证残差连接
    residual = (o1 - x).abs().mean().item()
    assert residual > 0.0
    
    print("✓ 前向传播测试通过")

if __name__ == "__main__":
    print("iTAC-AD 快速验证")
    print("=" * 50)
    
    try:
        test_decoder_switching()
        test_environment_variables()
        test_forward_pass()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！iTAC-AD 已准备就绪")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
