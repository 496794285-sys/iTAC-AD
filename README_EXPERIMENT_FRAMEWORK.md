# iTAC-AD 实验框架

这是一个完整的iTAC-AD异常检测实验框架，基于Tran-AD和iTransformer构建。

## 项目结构

```
iTAC-AD/
├── itac_eval/           # 评测模块
│   ├── __init__.py
│   └── metrics.py       # POT阈值、事件级F1、PR-AUC等评测指标
├── scripts/             # 实验脚本
│   ├── eval_event.sh    # 事件级评测脚本
│   ├── bench_one.sh     # 单次实验脚本
│   └── bench_grid.sh    # 网格实验脚本
├── tools/               # 分析工具
│   ├── aggregate_results.py  # 结果汇总
│   ├── topk_segments.py     # Top-K误报/漏报分析
│   ├── aac_sensitivity.py   # AAC超参敏感性分析
│   ├── make_tables.py       # 生成汇总表
│   └── make_figs.py         # 生成图表
├── tests/               # 单元测试
│   ├── test_pot_monotonic.py
│   ├── test_event_merge.py
│   └── test_grl_sign.py
├── main.py              # 主训练脚本
└── results/             # 实验结果
    └── all.csv          # 汇总结果
```

## 快速开始

### 1. 激活环境
```bash
conda activate itacad
```

### 2. 运行单次实验
```bash
MODEL=iTAC_AD DATASET=synthetic SEED=0 ./scripts/bench_one.sh
```

### 3. 运行网格实验
```bash
DATASETS=("synthetic") SEEDS=(0) bash scripts/bench_grid.sh
```

### 4. 生成汇总表
```bash
python tools/make_tables.py --csv results/all.csv
```

### 5. 分析误报/漏报
```bash
python tools/topk_segments.py --run vendor/tranad/outputs/latest --k 10
```

## 评测指标

- **POT阈值**: 基于极值理论的阈值估计
- **事件级F1**: IoU≥0.1的事件匹配F1分数
- **PR-AUC**: Precision-Recall曲线下面积
- **Precision/Recall**: 事件级精确率和召回率

## 实验配置

### 环境变量
- `ITAC_FORCE_CPU`: 强制使用CPU (默认: 1)
- `ITAC_EPOCHS`: 训练轮数
- `ITAC_MAX_STEPS`: 每轮最大步数
- `AAC_ALPHA`: AAC调度器α参数
- `AAC_BETA`: AAC调度器β参数
- `AAC_TAU`: AAC调度器τ参数
- `ITAC_GRL_LAMBDA`: GRL权重

### 模型配置
- `MODEL`: 模型类型 (iTAC_AD)
- `DATASET`: 数据集 (synthetic)
- `SEED`: 随机种子

## 实验结果

当前在synthetic数据集上的结果：
- F1: 0.64
- Precision: 1.0
- Recall: 0.47
- AUC-PR: 0.93

## 单元测试

```bash
python -m pytest tests/ -v
```

## 下一步计划

1. 扩展数据集支持 (SMD, MSL, SMAP等)
2. 实现TranAD和iT_BASE基线模型
3. 运行完整的消融实验
4. AAC超参敏感性分析
5. 性能分析和优化
