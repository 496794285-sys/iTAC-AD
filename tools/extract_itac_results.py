# -*- coding: utf-8 -*-
"""
提取 iTAC-AD 在 SMAP、MSL、SMD 数据集上的结果
"""
import json
import pandas as pd
import os

def main():
    # 定义结果文件路径
    results = [
        {
            "dataset": "SMAP",
            "model": "iTAC_AD", 
            "seed": 0,
            "run_dir": "vendor/tranad/outputs/20250917-154736-itac",
            "metrics_file": "vendor/tranad/outputs/20250917-154736-itac/metrics.json"
        },
        {
            "dataset": "MSL",
            "model": "iTAC_AD",
            "seed": 0, 
            "run_dir": "vendor/tranad/outputs/20250917-154825-itac",
            "metrics_file": "vendor/tranad/outputs/20250917-154825-itac/metrics.json"
        },
        {
            "dataset": "SMD",
            "model": "iTAC_AD",
            "seed": 0,
            "run_dir": "vendor/tranad/outputs/20250917-154856-itac", 
            "metrics_file": "vendor/tranad/outputs/20250917-154856-itac/metrics.json"
        }
    ]
    
    rows = []
    for result in results:
        try:
            # 读取 metrics.json
            with open(result["metrics_file"], 'r') as f:
                metrics = json.load(f)
            
            # 构建行数据
            row = {
                'model': result["model"],
                'dataset': result["dataset"],
                'seed': result["seed"],
                'run_dir': result["run_dir"],
                'f1': metrics.get('f1'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'auc_pr': metrics.get('auc_pr'),
                'auc_roc': metrics.get('auc_roc'),
                'threshold': metrics.get('threshold')
            }
            rows.append(row)
            print(f"成功读取 {result['dataset']}: F1={row['f1']:.4f}, P={row['precision']:.4f}, R={row['recall']:.4f}")
            
        except Exception as e:
            print(f"警告: 无法处理 {result['dataset']}: {e}")
            continue
    
    # 创建 DataFrame 并保存
    df = pd.DataFrame(rows)
    
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/itac_results.csv", index=False)
    
    print(f"\n提取了 {len(rows)} 个结果到 results/itac_results.csv")
    print("\n结果汇总:")
    print(df[['dataset', 'precision', 'recall', 'f1', 'auc_pr']].to_string(index=False))

if __name__ == "__main__":
    main()
