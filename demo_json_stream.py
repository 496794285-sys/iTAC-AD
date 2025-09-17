#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAC-AD JSON流处理功能快速演示
"""
import json
import numpy as np
import subprocess
import tempfile
import os

def demo_json_stream():
    """演示JSON流处理功能"""
    print("🚀 iTAC-AD JSON流处理功能演示")
    print("=" * 50)
    
    # 检查模型路径
    ckpt_dir = "/Users/waba/PythonProject/Transformer Project/iTAC-AD/release_v0.1.0/ckpt"
    if not os.path.exists(ckpt_dir):
        print(f"❌ 模型目录不存在: {ckpt_dir}")
        return
    
    print("📊 创建演示数据...")
    
    # 创建演示数据
    demo_jsonl = tempfile.mktemp(suffix='.jsonl')
    with open(demo_jsonl, "w") as f:
        np.random.seed(42)
        for t in range(50):
            # 生成5维传感器数据
            v = np.random.randn(5).tolist()
            # 在20-25时间点注入异常
            if 20 <= t <= 25:
                v = (np.array(v) + 2.0).tolist()
            
            data = {
                "timestamp": t,
                "sensor_id": f"sensor_{t%3}",
                "values": v,
                "metadata": {"location": "room_a", "type": "environmental"}
            }
            f.write(json.dumps(data) + "\n")
    
    print(f"✅ 演示数据已创建: {demo_jsonl}")
    print("   包含50个样本，其中20-25为异常段")
    
    # 运行JSON流处理
    cmd = [
        "itacad", "stream-json",
        "--ckpt", ckpt_dir,
        "--L", "10",
        "--jsonl", demo_jsonl,
        "--vector-field", "values",
        "--pot-q", "0.9",
        "--pot-level", "0.95"
    ]
    
    print("\n🔍 运行异常检测...")
    print("命令:", " ".join(cmd))
    print("\n输出结果:")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            # 解析并显示结果
            ready_event = None
            tick_events = []
            anomalies = []
            
            for line in lines:
                try:
                    data = json.loads(line)
                    event_type = data.get("event")
                    
                    if event_type == "ready":
                        ready_event = data
                        print(f"✅ 系统就绪: 窗口大小={data.get('L')}, 维度={data.get('D')}")
                    elif event_type == "infer_dim":
                        print(f"📏 自动推断维度: {data.get('D')}")
                    elif event_type == "tick":
                        tick_events.append(data)
                        ts = data.get("timestamp", "N/A")
                        score = data.get("score", 0)
                        thr = data.get("thr", 0)
                        anom = data.get("anom", 0)
                        
                        status = "🚨 异常" if anom else "✅ 正常"
                        print(f"时间{ts:2d}: 分数={score:.3f}, 阈值={thr:.3f} {status}")
                        
                        if anom:
                            anomalies.append(data)
                    elif event_type == "error":
                        print(f"❌ 错误: {data.get('msg')}")
                    elif event_type == "skip":
                        print(f"⏭️ 跳过: {data.get('reason')}")
                        
                except json.JSONDecodeError:
                    continue
            
            print("-" * 50)
            print(f"📈 统计结果:")
            print(f"   - 处理事件: {len(tick_events)}")
            print(f"   - 检测异常: {len(anomalies)}")
            
            if anomalies:
                anomaly_times = [e.get("timestamp") for e in anomalies]
                print(f"   - 异常时间点: {anomaly_times}")
                
                # 检查是否检测到了预期的异常段
                expected_anomalies = [t for t in anomaly_times if 20 <= t <= 25]
                if expected_anomalies:
                    print(f"   - 正确检测到预期异常: {len(expected_anomalies)}/{len(anomaly_times)}")
                else:
                    print("   - 未检测到预期异常段")
            
            print("\n🎉 演示完成！")
            
        else:
            print(f"❌ 命令执行失败:")
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
    finally:
        # 清理演示文件
        if os.path.exists(demo_jsonl):
            os.remove(demo_jsonl)
            print(f"🧹 演示文件已清理")

if __name__ == "__main__":
    demo_json_stream()
