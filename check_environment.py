#!/usr/bin/env python3
"""
iTAC-AD 环境检查脚本
检查必要的依赖是否已安装
"""
import sys

def check_dependencies():
    """检查必要的依赖"""
    missing = []
    
    # 检查 PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} 已安装")
        if torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon GPU) 支持可用")
        elif torch.cuda.is_available():
            print("✓ CUDA GPU 支持可用")
        else:
            print("ℹ 使用 CPU 模式")
    except ImportError:
        missing.append("torch")
        print("❌ PyTorch 未安装")
    
    # 检查其他依赖
    deps = ["numpy", "pandas", "scikit-learn", "matplotlib", "tqdm"]
    for dep in deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {dep} {version} 已安装")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep} 未安装")
    
    return missing

def print_installation_guide():
    """打印安装指南"""
    print("\n" + "="*60)
    print("📋 环境设置指南")
    print("="*60)
    print()
    print("1. 安装 PyTorch (推荐使用 conda):")
    print("   conda install pytorch torchvision torchaudio -c pytorch")
    print()
    print("2. 或者使用 pip:")
    print("   pip install torch torchvision torchaudio")
    print()
    print("3. 安装其他依赖:")
    print("   pip install numpy pandas scikit-learn matplotlib tqdm")
    print()
    print("4. 对于 Apple Silicon Mac (M1/M2):")
    print("   conda install pytorch torchvision torchaudio -c pytorch")
    print("   # 确保使用支持 MPS 的版本")
    print()
    print("5. 验证安装:")
    print("   python -c \"import torch; print('PyTorch version:', torch.__version__)\"")

if __name__ == "__main__":
    print("iTAC-AD 环境检查")
    print("="*40)
    
    missing = check_dependencies()
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print_installation_guide()
        sys.exit(1)
    else:
        print("\n🎉 所有依赖都已安装！")
        print("可以运行: python test_itac_ad_quick.py")
