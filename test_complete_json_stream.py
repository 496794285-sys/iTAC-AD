#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的JSON流处理功能测试
"""
import json
import numpy as np
import subprocess
import os
import tempfile

def test_vector_field():
    """测试vector-field方式"""
    print("🧪 测试 --vector-field 方式")
    
    # 创建测试数据
    test_jsonl = tempfile.mktemp(suffix='.jsonl')
    with open(test_jsonl, "w") as f:
        np.random.seed(42)
        for t in range(15):
            v = np.random.randn(5).tolist()
            data = {"ts": t, "values": v, "id": f"sample_{t}"}
            f.write(json.dumps(data) + "\n")
    
    cmd = [
        "itacad", "stream-json",
        "--ckpt", "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt",
        "--L", "8",
        "--jsonl", test_jsonl,
        "--vector-field", "values"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        success = result.returncode == 0
        if success:
            lines = result.stdout.strip().split('\n')
            tick_events = [json.loads(line) for line in lines if '"event": "tick"' in line]
            print(f"✅ 成功处理 {len(tick_events)} 个事件")
        else:
            print(f"❌ 失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        success = False
    finally:
        os.remove(test_jsonl)
    
    return success

def test_fields():
    """测试fields方式"""
    print("\n🧪 测试 --fields 方式")
    
    # 创建测试数据
    test_jsonl = tempfile.mktemp(suffix='.jsonl')
    with open(test_jsonl, "w") as f:
        np.random.seed(123)
        for t in range(15):
            data = {
                "ts": t,
                "temp": np.random.randn(),
                "press": np.random.randn(),
                "flow": np.random.randn(),
                "humidity": np.random.randn(),
                "voltage": np.random.randn()
            }
            f.write(json.dumps(data) + "\n")
    
    cmd = [
        "itacad", "stream-json",
        "--ckpt", "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt",
        "--L", "8",
        "--jsonl", test_jsonl,
        "--fields", "temp,press,flow,humidity,voltage"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        success = result.returncode == 0
        if success:
            lines = result.stdout.strip().split('\n')
            tick_events = [json.loads(line) for line in lines if '"event": "tick"' in line]
            print(f"✅ 成功处理 {len(tick_events)} 个事件")
        else:
            print(f"❌ 失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        success = False
    finally:
        os.remove(test_jsonl)
    
    return success

def test_prefix():
    """测试prefix方式"""
    print("\n🧪 测试 --prefix 方式")
    
    # 创建测试数据
    test_jsonl = tempfile.mktemp(suffix='.jsonl')
    with open(test_jsonl, "w") as f:
        np.random.seed(456)
        for t in range(15):
            data = {"ts": t}
            # 添加带前缀的特征
            for i in range(5):
                data[f"f_{i}"] = np.random.randn()
            f.write(json.dumps(data) + "\n")
    
    cmd = [
        "itacad", "stream-json",
        "--ckpt", "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt",
        "--L", "8",
        "--jsonl", test_jsonl,
        "--prefix", "f_"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        success = result.returncode == 0
        if success:
            lines = result.stdout.strip().split('\n')
            tick_events = [json.loads(line) for line in lines if '"event": "tick"' in line]
            print(f"✅ 成功处理 {len(tick_events)} 个事件")
        else:
            print(f"❌ 失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        success = False
    finally:
        os.remove(test_jsonl)
    
    return success

def test_stdin():
    """测试stdin输入"""
    print("\n🧪 测试 stdin 输入")
    
    cmd = [
        "itacad", "stream-json",
        "--ckpt", "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt",
        "--L", "5",
        "--vector-field", "values"
    ]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 发送测试数据
        np.random.seed(789)
        test_data = []
        for t in range(10):
            v = np.random.randn(3).tolist()
            data = {"ts": t, "values": v}
            test_data.append(json.dumps(data) + "\n")
        
        # 一次性发送所有数据
        input_data = "".join(test_data)
        stdout, stderr = process.communicate(input=input_data, timeout=10)
        
        success = process.returncode == 0
        if success:
            lines = stdout.strip().split('\n')
            tick_events = [json.loads(line) for line in lines if '"event": "tick"' in line]
            print(f"✅ 成功处理 {len(tick_events)} 个事件")
        else:
            print(f"❌ 失败: {stderr}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        success = False
    
    return success

def test_anomaly_detection():
    """测试异常检测"""
    print("\n🧪 测试异常检测")
    
    test_jsonl = tempfile.mktemp(suffix='.jsonl')
    with open(test_jsonl, "w") as f:
        np.random.seed(999)
        for t in range(30):
            v = np.random.randn(4).tolist()
            # 在10-15时间点注入异常
            if 10 <= t <= 15:
                v = (np.array(v) + 2.5).tolist()
            data = {"ts": t, "values": v}
            f.write(json.dumps(data) + "\n")
    
    cmd = [
        "itacad", "stream-json",
        "--ckpt", "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt",
        "--L", "8",
        "--jsonl", test_jsonl,
        "--vector-field", "values",
        "--pot-q", "0.9",
        "--pot-level", "0.95"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        success = result.returncode == 0
        if success:
            lines = result.stdout.strip().split('\n')
            tick_events = [json.loads(line) for line in lines if '"event": "tick"' in line]
            anomalies = [e for e in tick_events if e.get("anom", 0) == 1]
            print(f"✅ 处理了 {len(tick_events)} 个事件，检测到 {len(anomalies)} 个异常")
            if anomalies:
                anomaly_times = [e.get("ts") for e in anomalies]
                print(f"   异常时间点: {anomaly_times}")
        else:
            print(f"❌ 失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 错误: {e}")
        success = False
    finally:
        os.remove(test_jsonl)
    
    return success

if __name__ == "__main__":
    print("🚀 开始完整JSON流处理功能测试")
    print("=" * 60)
    
    tests = [
        test_vector_field,
        test_fields,
        test_prefix,
        test_stdin,
        test_anomaly_detection
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 出现异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！JSON流处理功能完全正常")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
