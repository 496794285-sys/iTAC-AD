# -*- coding: utf-8 -*-
"""
扫描 outputs 目录并聚合所有结果到 results/all.csv
"""
import argparse
import os
import json
import pandas as pd
import glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="outputs", help="扫描的目录")
    ap.add_argument("--out", default="results/all.csv", help="输出文件")
    args = ap.parse_args()

    rows = []
    
    # 扫描所有 outputs 子目录
    pattern = os.path.join(args.scan, "**", "metrics.json")
    metric_files = glob.glob(pattern, recursive=True)
    
    for metric_file in metric_files:
        try:
            # 读取 metrics.json
            with open(metric_file, 'r') as f:
                metrics = json.load(f)
            
            # 从路径推断信息
            parts = metric_file.split(os.sep)
            # 期望路径格式: outputs/{dataset}/{model}/.../metrics.json
            if len(parts) >= 3:
                dataset = parts[1]
                model = parts[2]
            else:
                continue
                
            # 尝试读取 config.yaml 获取更多信息
            config_file = os.path.join(os.path.dirname(metric_file), "config.yaml")
            seed = 0  # 默认值
            if os.path.exists(config_file):
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    seed = config.get('seed', 0)
                except:
                    pass
            
            # 构建行数据
            row = {
                'model': model,
                'dataset': dataset,
                'seed': seed,
                'run_dir': os.path.dirname(metric_file),
                'f1': metrics.get('f1'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'auc_pr': metrics.get('auc_pr'),
                'auc_roc': metrics.get('auc_roc'),
                'threshold': metrics.get('threshold')
            }
            rows.append(row)
            
        except Exception as e:
            print(f"警告: 无法处理 {metric_file}: {e}")
            continue
    
    if not rows:
        print("未找到任何结果文件")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(rows)
    
    # 保存结果
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    
    print(f"聚合了 {len(rows)} 个结果到 {args.out}")
    print("\n数据集统计:")
    print(df.groupby(['model', 'dataset']).size().reset_index(name='count'))

if __name__ == "__main__":
    main()
