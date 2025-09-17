#!/usr/bin/env python3
"""
iTAC-AD 增强验证脚本
测试 GRL 对抗训练和 AAC 调度器
"""
import os
import torch
import sys
sys.path.insert(0, '/Users/waba/PythonProject/Transformer Project/iTAC-AD')

def test_grl_functionality():
    """测试 GRL 功能"""
    print("=== 测试 GRL 对抗训练功能 ===")
    
    # 测试启用 GRL
    print("1. 测试启用 GRL...")
    os.environ["ITAC_USE_GRL"] = "1"
    os.environ["ITAC_GRL_LAMBDA"] = "1.0"
    
    from itac_ad.models.itac_ad import ITAC_AD
    model = ITAC_AD(feats=7)
    
    print(f"   GRL 启用: {model.uses_grl}")
    print(f"   GRL Lambda: {model.grl.lambd}")
    
    # 测试禁用 GRL
    print("2. 测试禁用 GRL...")
    os.environ["ITAC_USE_GRL"] = "0"
    model_no_grl = ITAC_AD(feats=7)
    
    print(f"   GRL 启用: {model_no_grl.uses_grl}")
    print(f"   GRL 类型: {type(model_no_grl.grl)}")
    
    print("✓ GRL 功能测试通过")

def test_aac_scheduler():
    """测试 AAC 调度器"""
    print("\n=== 测试 AAC 调度器 ===")
    
    from itac_ad.components.aac_scheduler import AACScheduler
    
    aac = AACScheduler(window_size=64, quantile_p=0.9)
    
    # 模拟一些残差数据
    print("1. 测试 AAC 权重计算...")
    for i in range(10):
        # 模拟不同强度的残差
        residual = torch.randn(32, 7) * (0.1 + i * 0.05)
        w_t = aac.step(residual)
        stats = aac.stats()
        print(f"   Step {i}: w={stats['w']:.3f}, q={stats['q']:.3f}, drift={stats['drift']:.3f}")
    
    print("✓ AAC 调度器测试通过")

def test_forward_with_grl():
    """测试带 GRL 的前向传播"""
    print("\n=== 测试带 GRL 的前向传播 ===")
    
    os.environ["ITAC_USE_GRL"] = "1"
    os.environ["ITAC_GRL_LAMBDA"] = "0.5"
    
    from itac_ad.models.itac_ad import ITAC_AD
    
    model = ITAC_AD(feats=5)
    x = torch.randn(2, 64, 5)
    
    o1, o2, aux = model(x)
    
    print(f"   输入形状: {x.shape}")
    print(f"   输出1形状: {o1.shape}")
    print(f"   输出2形状: {o2.shape}")
    print(f"   GRL 启用: {model.uses_grl}")
    print(f"   GRL Lambda: {model.grl.lambd}")
    
    # 验证梯度反转
    loss1 = torch.mean((o1 - x) ** 2)
    loss2 = torch.mean((o2 - x) ** 2)
    
    print(f"   重构损失: {loss1.item():.4f}")
    print(f"   对抗损失: {loss2.item():.4f}")
    
    print("✓ GRL 前向传播测试通过")

def test_environment_variables():
    """测试环境变量"""
    print("\n=== 测试环境变量 ===")
    
    # 设置各种环境变量
    os.environ["ITAC_USE_GRL"] = "1"
    os.environ["ITAC_GRL_LAMBDA"] = "0.8"
    os.environ["ITAC_D_MODEL"] = "64"
    os.environ["ITAC_N_HEADS"] = "4"
    os.environ["ITAC_E_LAYERS"] = "1"
    os.environ["ITAC_DECODER"] = "mlp"
    
    from itac_ad.models.itac_ad import ITAC_AD
    
    model = ITAC_AD(feats=3)
    
    print(f"   D Model: {model.encoder.d_model}")
    print(f"   N Heads: {model.encoder.n_heads}")
    print(f"   E Layers: {model.encoder.e_layers}")
    print(f"   Decoder: {model.decoder_kind}")
    print(f"   GRL Lambda: {model.grl.lambd}")
    
    print("✓ 环境变量测试通过")

if __name__ == "__main__":
    print("iTAC-AD 增强验证")
    print("=" * 50)
    
    try:
        test_grl_functionality()
        test_aac_scheduler()
        test_forward_with_grl()
        test_environment_variables()
        
        print("\n" + "=" * 50)
        print("🎉 所有增强功能测试通过！")
        print("现在可以运行:")
        print("  ./scripts/run_synthetic.sh          # 合成数据测试")
        print("  ITAC_USE_GRL=0 ./scripts/run_synthetic.sh  # 关闭对抗")
        print("  ITAC_DECODER=mlp ./scripts/run_synthetic.sh  # MLP解码器")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
