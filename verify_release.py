#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAC-AD v0.1.0 发布验证脚本
验证所有发布功能是否正常工作
"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from pathlib import Path

def verify_installation():
    """验证包安装"""
    print("🔍 验证包安装...")
    try:
        result = subprocess.run(['itacad', '--help'], capture_output=True, text=True)
        if result.returncode == 0 and 'iTAC-AD CLI' in result.stdout:
            print("✅ 包安装成功")
            return True
        else:
            print("❌ 包安装失败")
            return False
    except Exception as e:
        print(f"❌ 包安装异常: {e}")
        return False

def verify_batch_inference():
    """验证批量推理功能"""
    print("🔍 验证批量推理功能...")
    try:
        # 创建测试数据
        np.random.seed(42)
        data = np.random.randn(30, 7)
        data[10:15, :] += 2.0  # 添加异常
        
        labels = np.zeros(30)
        labels[10:15] = 1
        
        df = pd.DataFrame(data, columns=[f'f{i}' for i in range(7)])
        df['label'] = labels
        df.to_csv('test_release.csv', index=False)
        
        # 运行推理
        result = subprocess.run([
            'itacad', 'predict',
            '--csv', 'test_release.csv',
            '--ckpt', 'vendor/tranad/outputs/20250917-110600-itac',
            '--window', '10',
            '--label-col', 'label',
            '--out', 'outputs/release_test'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists('outputs/release_test/metrics.json'):
            print("✅ 批量推理功能正常")
            return True
        else:
            print("❌ 批量推理功能异常")
            return False
    except Exception as e:
        print(f"❌ 批量推理异常: {e}")
        return False

def verify_stream_processing():
    """验证实时流处理功能"""
    print("🔍 验证实时流处理功能...")
    try:
        # 创建测试流
        def generate_stream():
            np.random.seed(0)
            for t in range(12):
                x = np.random.randn(7)
                if 5<=t<=7: x += 2.0
                yield ','.join(f'{v:.4f}' for v in x)
        
        stream_data = '\n'.join(generate_stream())
        result = subprocess.run([
            'itacad', 'stream',
            '--ckpt', 'vendor/tranad/outputs/20250917-110600-itac',
            '--L', '10',
            '--D', '7'
        ], input=stream_data, capture_output=True, text=True)
        
        if result.returncode == 0 and 'ready' in result.stdout:
            print("✅ 实时流处理功能正常")
            return True
        else:
            print("❌ 实时流处理功能异常")
            return False
    except Exception as e:
        print(f"❌ 实时流处理异常: {e}")
        return False

def verify_cli_commands():
    """验证CLI命令"""
    print("🔍 验证CLI命令...")
    commands = [
        ['itacad', '--help'],
        ['itacad', 'predict', '--help'],
        ['itacad', 'stream', '--help'],
        ['itacad', 'export', '--help'],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ CLI命令失败: {' '.join(cmd)}")
                return False
        except Exception as e:
            print(f"❌ CLI命令异常: {' '.join(cmd)} - {e}")
            return False
    
    print("✅ CLI命令正常")
    return True

def verify_file_structure():
    """验证文件结构"""
    print("🔍 验证文件结构...")
    required_files = [
        'itacad/cli.py',
        'itacad/infer/predict.py',
        'itacad/export.py',
        'rt/stream_runner.py',
        'pyproject.toml',
        'README.md',
        'LICENSE',
        'CITATION.cff',
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少文件: {file_path}")
            return False
    
    print("✅ 文件结构完整")
    return True

def main():
    """主验证函数"""
    print("🚀 iTAC-AD v0.1.0 发布验证")
    print("=" * 50)
    
    tests = [
        ("包安装", verify_installation),
        ("文件结构", verify_file_structure),
        ("CLI命令", verify_cli_commands),
        ("批量推理", verify_batch_inference),
        ("实时流处理", verify_stream_processing),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 验证: {name}")
        success = test_func()
        results.append((name, success))
    
    print("\n" + "=" * 50)
    print("📊 验证结果汇总:")
    
    passed = 0
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 总计: {passed}/{len(results)} 个验证通过")
    
    if passed == len(results):
        print("🎉 所有功能验证通过！项目已准备好发布！")
        return True
    else:
        print("⚠️  部分功能需要修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
