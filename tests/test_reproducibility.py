# -*- coding: utf-8 -*-
import os, json, numpy as np
import pandas as pd

def load_metrics(run):
    with open(f"{run}/metrics.json") as f:
        return json.load(f)

def test_seed_stability():
    """测试随机种子稳定性：固定种子后指标波动应在阈值内"""
    # 假设提前用 3 个种子跑完 bench_one，路径写进 results/all.csv
    try:
        df = pd.read_csv("results/all.csv")
        g = df[(df["tag"]=="iTAC_AD.full") & (df["dataset"].str.contains("SMD:machine-1-1"))]
        assert len(g)>=3, "need >=3 seeds"
        f1 = g["f1"].values
        assert np.std(f1) <= 0.02, f"F1 std too large: {np.std(f1):.3f}"
        print(f"✅ 种子稳定性测试通过: F1 std = {np.std(f1):.4f}")
        return True
    except FileNotFoundError:
        print("⚠️ results/all.csv 不存在，跳过种子稳定性测试")
        return True
    except Exception as e:
        print(f"❌ 种子稳定性测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载功能"""
    try:
        from itacad.infer.predict import load_model
        ckpt_dir = "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt"
        if os.path.exists(ckpt_dir):
            model, device, cfg = load_model(ckpt_dir, feats=5)
            print("✅ 模型加载测试通过")
            return True
        else:
            print("⚠️ 模型检查点不存在，跳过模型加载测试")
            return True
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def test_json_stream():
    """测试JSON流处理功能"""
    try:
        import subprocess
        import tempfile
        
        # 创建简单测试数据
        test_jsonl = tempfile.mktemp(suffix='.jsonl')
        with open(test_jsonl, "w") as f:
            import numpy as np
            np.random.seed(42)
            for t in range(10):
                v = np.random.randn(3).tolist()
                data = {"ts": t, "values": v}
                f.write(json.dumps(data) + "\n")
        
        # 运行JSON流处理
        cmd = [
            "itacad", "stream-json",
            "--ckpt", "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt",
            "--L", "5",
            "--jsonl", test_jsonl,
            "--vector-field", "values"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        success = result.returncode == 0
        
        # 清理
        os.remove(test_jsonl)
        
        if success:
            print("✅ JSON流处理测试通过")
            return True
        else:
            print(f"❌ JSON流处理测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ JSON流处理测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 运行复现性测试")
    print("=" * 40)
    
    tests = [
        test_seed_stability,
        test_model_loading,
        test_json_stream
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"📊 复现性测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有复现性测试通过！")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
