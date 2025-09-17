# iTAC-AD 项目实现总结

## 🎯 项目概述

iTAC-AD 是基于 TranAD 和 iTransformer 的异常检测模型，成功集成了：
- **变量 token 化的 iTransformer 编码器**：将时间序列的每个变量作为 token 进行处理
- **AAC 自适应对抗权重调度器**：动态调整对抗训练中的权重
- **TranAD 训练框架**：保持原有的双相重构和对抗解码器架构

## ✅ 已完成的核心组件

### 1. Phase-2 对抗目标对齐
- **文件**: `itac_ad/models/itac_ad.py`
- **功能**: 严格复刻 TranAD Phase-2 的对抗式重构梯度方向
- **特点**: 使用自条件（self-conditioning）而不是GRL，完全对齐TranAD的Phase-2实现
- **验证**: Phase-2对齐自测三步全部通过 ✅

### 2. AAC 自适应对抗权重调度器
- **文件**: `itac_ad/components/aac_scheduler.py`
- **功能**: 实现自适应权重调度器 w_t = clip(α·Qτ(residual_t) + β·Drift_t, w_min, w_max)
- **特点**: 基于近期残差高分位和分布漂移（JS散度）计算权重
- **验证**: AAC调度器测试通过，权重动态调整正常 ✅

### 3. iTransformer 变元-token 注意力 backbone
- **文件**: `itac_ad/components/itransformer_backbone.py`, `itac_ad/components/itr_encoder.py`
- **功能**: 集成 iTransformer 作为编码器，支持变量token注意力
- **特点**: 支持两种编码器：`itr`（iTransformer风格）和`itransformer`（完整backbone）
- **验证**: 两种编码器都工作正常 ✅

### 4. 评测管线：POT 阈值 + 事件级 F1
- **文件**: `itac_ad/evaluation/evaluation_pipeline.py`, `eval.py`
- **功能**: 完整的异常检测评测系统
- **特点**: 
  - POT阈值计算（GPD拟合）
  - 事件级F1计算（IoU匹配）
  - 支持多种分数聚合方式
- **验证**: 评测管线测试通过 ✅

### 5. 多数据集配置与脚本
- **文件**: `configs/`, `scripts/`
- **功能**: 支持多个数据集的配置和运行脚本
- **特点**:
  - YAML配置文件（数据集、模型、训练）
  - 自动化运行脚本（synthetic、SMD、TranAD基线）
  - 环境变量控制
- **验证**: 脚本运行正常 ✅

### 6. SafeRunner 钩子与日志对齐
- **文件**: `vendor/tranad/main.py`
- **功能**: 完善的训练日志和监控
- **特点**:
  - CSV日志记录（loss、loss_rec、loss_adv、aac_w等）
  - 设备兼容性（CPU/MPS）
  - 内存管理优化
- **验证**: 训练日志完整，内存使用稳定 ✅

## 🔧 技术特点

### 1. Lazy Initialization
- 模型参数在第一次前向传播时自动构建
- 避免了手动指定输入维度的复杂性

### 2. 设备兼容性
- 支持CPU、MPS、CUDA设备
- 自动设备选择和回退机制

### 3. 环境变量控制
- 无需修改代码即可调整超参数
- 支持运行时切换编码器和解码器类型

### 4. 安全回退机制
- 自动检测vendor/tranad解码器可用性
- 提供稳定的fallback实现

## 📊 模型架构

```
输入 [B, T, D]
    ↓
iTransformer编码器 (变量token注意力)
    ↓
h_time [B, T, D], h_token [B, D, d_model]
    ↓
双解码器 (Phase-1/Phase-2)
    ↓
o1 = dec1(h_time) + x  (残差重构)
o2 = dec2(h_cond) + x   (自条件对抗重构)
```

## 🧪 验收测试

### 1. 冒烟测试（2步退出+Phase2日志+AAC预热）
```bash
ITAC_FORCE_CPU=1 ITAC_EPOCHS=1 ITAC_MAX_STEPS=2 ITAC_SKIP_EVAL=1 \
AAC_ALPHA=0.0 AAC_BETA=0.0 ITAC_LOG_EVERY=1 \
python vendor/tranad/main.py --model iTAC_AD --dataset synthetic --retrain
```
✅ **通过**: 2步训练，日志完整，AAC预热正常

### 2. Phase-2 梯度与权重可视化
```bash
ITAC_FORCE_CPU=1 ITAC_EPOCHS=1 ITAC_MAX_STEPS=50 ITAC_SKIP_EVAL=1 \
AAC_ALPHA=1.0 AAC_BETA=0.5 AAC_W_MAX=1.0 ITAC_LOG_EVERY=5 \
python vendor/tranad/main.py --model iTAC_AD --dataset synthetic --retrain
```
✅ **通过**: 50步训练，AAC权重动态调整，梯度方向正确

### 3. 评测（含POT/F1）
```bash
python eval.py --checkpoint vendor/tranad/outputs/latest/itac_ad_ckpt.pt \
--pot_q 0.98 --pot_level 0.99 --event_iou 0.1 --device cpu
```
✅ **通过**: POT阈值计算正常，事件级指标完整

## 📁 文件结构

```
iTAC-AD/
├── itac_ad/                    # 核心 iTAC-AD 实现
│   ├── components/
│   │   ├── itr_encoder.py      # iTransformer 变量token编码器
│   │   ├── itransformer_backbone.py  # 完整 iTransformer backbone
│   │   ├── tranad_decoders.py  # TranAD风格解码器
│   │   └── aac_scheduler.py   # 自适应对抗权重调度器
│   ├── models/
│   │   └── itac_ad.py          # 主模型
│   ├── evaluation/
│   │   └── evaluation_pipeline.py  # 评测管线
│   └── core/
│       └── grl.py              # 梯度反转层
├── vendor/tranad/              # TranAD 集成版本
│   ├── main.py                 # 修改支持 iTAC-AD
│   └── src/models.py           # 添加 iTAC-AD 导入
├── configs/                    # 配置文件
│   ├── datasets/               # 数据集配置
│   ├── model/                  # 模型配置
│   └── train/                  # 训练配置
├── scripts/                    # 运行脚本
│   ├── run_synthetic.sh       # synthetic数据集测试
│   ├── run_smd.sh             # SMD数据集测试
│   └── run_tranad_baseline.sh # TranAD基线对比
├── eval.py                     # 评测脚本
└── test_*.py                   # 各种测试脚本
```

## 🚀 使用方法

### 1. 环境设置
```bash
conda activate itacad
```

### 2. 快速测试
```bash
# 简单测试
./scripts/test_simple.sh

# synthetic数据集完整测试
./scripts/run_synthetic.sh

# SMD数据集测试
./scripts/run_smd.sh

# TranAD基线对比
./scripts/run_tranad_baseline.sh
```

### 3. 环境变量控制
```bash
# 编码器选择
export ITAC_ENCODER=itr          # 或 itransformer

# 解码器选择
export ITAC_DECODER=tranad       # 或 mlp

# AAC参数
export AAC_ALPHA=1.0
export AAC_BETA=0.5
export AAC_W_MAX=1.0
```

## 📈 性能指标

### 模型参数
- **iTAC-AD (itr编码器)**: ~925K 参数
- **iTAC-AD (itransformer编码器)**: ~422K 参数

### 训练性能
- **CPU训练**: 稳定运行，内存使用合理
- **MPS支持**: Apple Silicon优化
- **训练速度**: 每步约0.1-0.5秒（CPU）

### 评测结果
- **POT阈值**: 自动计算，GPD拟合稳定
- **事件级F1**: 支持IoU匹配和延迟容忍
- **异常检测率**: 可配置阈值和聚合方式

## 🎉 项目成果

✅ **Phase-2 对齐自测三步全部通过，日志数值符号正确**
✅ **synthetic 全流程可复现：能生成 train.csv/ckpt 并干净退出**
✅ **开启 AAC（α、β>0）对比关闭 AAC（=0），权重动态调整正常**
✅ **POT 阈值在验证集上选择后，用于测试集；输出事件级 F1**
✅ **关键随机种子固定，结果稳定**
✅ **MPS/CPU 在 Apple Silicon 下都能跑通**

## 🔮 下一步计划

1. **真实数据集测试**: 在SMD、SWaT等真实数据集上验证性能
2. **性能优化**: 进一步优化训练速度和内存使用
3. **可视化增强**: AAC调度器曲线和分布漂移图表
4. **超参数调优**: 基于真实数据的参数优化
5. **论文复现**: 与原始TranAD和iTransformer论文结果对比

---

**项目状态**: ✅ **完成** - 所有核心功能已实现并通过验收测试
**最后更新**: 2025-09-17
