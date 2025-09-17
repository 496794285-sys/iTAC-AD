#!/usr/bin/env python3
"""
iTAC-AD 训练日志分析脚本
分析 w/q/drift 的走向和趋势
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_training_log(csv_path):
    """分析训练日志"""
    print("📊 iTAC-AD 训练日志分析")
    print("=" * 50)
    
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    print(f"📁 日志文件: {csv_path}")
    print(f"📈 总步数: {len(df)}")
    print(f"🔄 训练轮数: {df['epoch'].min()} - {df['epoch'].max()}")
    print()
    
    # 基本统计信息
    print("📊 基本统计信息:")
    print(f"   损失范围: {df['loss'].min():.4f} - {df['loss'].max():.4f}")
    print(f"   AAC权重(w): {df['w'].min():.3f} - {df['w'].max():.3f}")
    print(f"   残差高分位(q): {df['q'].min():.3f} - {df['q'].max():.3f}")
    print(f"   分布漂移(drift): {df['drift'].min():.3f} - {df['drift'].max():.3f}")
    print()
    
    # 趋势分析
    print("📈 趋势分析:")
    print(f"   损失变化: {df['loss'].iloc[0]:.4f} → {df['loss'].iloc[-1]:.4f} ({((df['loss'].iloc[-1]/df['loss'].iloc[0]-1)*100):+.1f}%)")
    print(f"   AAC权重变化: {df['w'].iloc[0]:.3f} → {df['w'].iloc[-1]:.3f} ({((df['w'].iloc[-1]/df['w'].iloc[0]-1)*100):+.1f}%)")
    print(f"   残差高分位变化: {df['q'].iloc[0]:.3f} → {df['q'].iloc[-1]:.3f} ({((df['q'].iloc[-1]/df['q'].iloc[0]-1)*100):+.1f}%)")
    print(f"   分布漂移变化: {df['drift'].iloc[0]:.3f} → {df['drift'].iloc[-1]:.3f} ({((df['drift'].iloc[-1]/df['drift'].iloc[0]-1)*100):+.1f}%)")
    print()
    
    # 创建可视化图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('iTAC-AD 训练过程分析', fontsize=16, fontweight='bold')
    
    # 1. 损失曲线
    axes[0,0].plot(df.index, df['loss'], 'b-', linewidth=2, label='Loss')
    axes[0,0].set_title('训练损失变化')
    axes[0,0].set_xlabel('训练步数')
    axes[0,0].set_ylabel('损失值')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. AAC权重(w)变化
    axes[0,1].plot(df.index, df['w'], 'r-', linewidth=2, label='AAC Weight (w)')
    axes[0,1].set_title('AAC权重(w)变化')
    axes[0,1].set_xlabel('训练步数')
    axes[0,1].set_ylabel('权重值')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. 残差高分位(q)变化
    axes[1,0].plot(df.index, df['q'], 'g-', linewidth=2, label='Quantile (q)')
    axes[1,0].set_title('残差高分位(q)变化')
    axes[1,0].set_xlabel('训练步数')
    axes[1,0].set_ylabel('高分位值')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 4. 分布漂移(drift)变化
    axes[1,1].plot(df.index, df['drift'], 'm-', linewidth=2, label='Drift (z)')
    axes[1,1].set_title('分布漂移(drift)变化')
    axes[1,1].set_xlabel('训练步数')
    axes[1,1].set_ylabel('漂移值')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = os.path.dirname(csv_path)
    plot_path = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 分析图表已保存: {plot_path}")
    
    # 显示图表
    plt.show()
    
    # AAC 调度器行为分析
    print("\n🔍 AAC 调度器行为分析:")
    print("=" * 30)
    
    # 计算相关系数
    corr_loss_w = df['loss'].corr(df['w'])
    corr_loss_q = df['loss'].corr(df['q'])
    corr_loss_drift = df['loss'].corr(df['drift'])
    
    print(f"   损失 vs AAC权重: {corr_loss_w:.3f}")
    print(f"   损失 vs 残差高分位: {corr_loss_q:.3f}")
    print(f"   损失 vs 分布漂移: {corr_loss_drift:.3f}")
    print()
    
    # 分析 AAC 调度器的响应
    w_std = df['w'].std()
    q_std = df['q'].std()
    drift_std = df['drift'].std()
    
    print("📊 AAC 调度器稳定性:")
    print(f"   AAC权重标准差: {w_std:.3f} (越小越稳定)")
    print(f"   残差高分位标准差: {q_std:.3f}")
    print(f"   分布漂移标准差: {drift_std:.3f}")
    print()
    
    # 建议
    print("💡 分析建议:")
    print("=" * 20)
    
    if corr_loss_q < -0.5:
        print("   ✅ 残差高分位与损失负相关，说明重构质量在提升")
    elif corr_loss_q > 0.5:
        print("   ⚠️  残差高分位与损失正相关，可能需要调整学习率")
    else:
        print("   ℹ️  残差高分位与损失相关性较弱")
    
    if w_std < 0.05:
        print("   ✅ AAC权重稳定，调度器工作良好")
    else:
        print("   ⚠️  AAC权重波动较大，可能需要调整调度器参数")
    
    if df['q'].iloc[-1] < df['q'].iloc[0] * 0.8:
        print("   ✅ 残差高分位显著下降，模型重构能力提升")
    else:
        print("   ℹ️  残差高分位变化不大，可能需要更多训练")
    
    return df

if __name__ == "__main__":
    # 使用最新的训练日志
    csv_path = "/Users/waba/PythonProject/Transformer Project/iTAC-AD/vendor/tranad/outputs/20250916-145017-itac/train.csv"
    
    if os.path.exists(csv_path):
        df = analyze_training_log(csv_path)
    else:
        print(f"❌ 日志文件不存在: {csv_path}")
        print("请先运行训练脚本生成日志文件")
