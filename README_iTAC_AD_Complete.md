# iTAC-AD 项目总结与使用指南

## 🎯 项目概述

iTAC-AD 是基于 Tran-AD 和 iTransformer 的异常检测模型，采用变量token编码器 + 双解码器架构。

## 📁 项目结构

```
iTAC-AD/
├── itac_ad/                    # 核心 iTAC-AD 实现
│   ├── components/
│   │   ├── itr_encoder.py      # iTransformer 变量token编码器
│   │   ├── tranad_decoders.py  # TranAD风格解码器（vendor优先+fallback）
│   │   └── aac_scheduler.py   # 自适应异常分数调度器
│   └── models/
│       └── itac_ad.py          # 主模型（支持解码器切换）
├── vendor/tranad/              # Tran-AD 参考实现（保持不变）
├── configs/                    # 配置文件
│   ├── model/itac_ad.yaml     # 模型配置
│   ├── dataset/synthetic.yaml # 数据集配置
│   └── train/light.yaml       # 训练配置
├── scripts/                    # 一键运行脚本
│   ├── run_synthetic.sh       # iTAC-AD 快速测试
│   └── run_tranad_baseline.sh # TranAD 基线对比
├── tests/                      # 单元测试
│   ├── test_import.py         # 导入测试
│   ├── test_forward.py        # 前向传播测试
│   └── test_mini_epoch.py     # 迷你训练测试
└── check_environment.py       # 环境检查脚本
```

## 🚀 快速开始

### 1. 环境检查
```bash
python check_environment.py
```

### 2. 安装依赖（如果缺少）
```bash
# 安装 PyTorch (Apple Silicon)
conda install pytorch torchvision torchaudio -c pytorch

# 安装其他依赖
pip install scikit-learn
```

### 3. 运行测试
```bash
# 快速验证
python test_itac_ad_quick.py

# 运行单元测试
pytest tests/ -v
```

### 4. 运行实验
```bash
# iTAC-AD 模型（TranAD解码器）
./scripts/run_synthetic.sh

# iTAC-AD 模型（MLP解码器）
ITAC_DECODER=mlp ./scripts/run_synthetic.sh

# TranAD 基线对比
./scripts/run_tranad_baseline.sh
```

## ⚙️ 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ITAC_DECODER` | `tranad` | 解码器类型：`mlp` 或 `tranad` |
| `ITAC_D_MODEL` | `128` | 编码器隐藏维度 |
| `ITAC_N_HEADS` | `8` | 注意力头数 |
| `ITAC_E_LAYERS` | `2` | 编码器层数 |
| `ITAC_DROPOUT` | `0.1` | Dropout率 |
| `ITAC_DEC_DEPTH` | `2` | TranAD解码器深度 |
| `ITAC_DEC_HEADS` | `4` | TranAD解码器头数 |
| `ITAC_DEC_DROPOUT` | `0.1` | TranAD解码器Dropout |

## 🔧 核心特性

### 1. 解码器切换
- **MLP解码器**：简单的两层线性+残差
- **TranAD解码器**：优先使用vendor实现，fallback到时间域Transformer

### 2. 环境变量控制
- 无需修改代码即可调整超参数
- 支持运行时切换解码器类型

### 3. 安全回退机制
- 自动检测vendor/tranad解码器可用性
- 提供稳定的fallback实现

### 4. 兼容性设计
- 保持与TranAD的接口兼容
- 支持现有数据处理流程

## 📊 模型架构

```
输入 [B, T, D]
    ↓
VariateTokenEncoder (iTransformer风格)
    ↓
h_time [B, T, D], h_token [B, D, d_model]
    ↓
双解码器 (Phase-1/Phase-2)
    ↓
o1 = dec1(h_time) + x  (残差重构)
o2 = dec2(h_time)      (对抗路)
```

## 🧪 测试覆盖

- ✅ 导入测试：验证模块正确导入
- ✅ 形状测试：验证输入输出形状
- ✅ 前向传播：验证模型前向计算
- ✅ 迷你训练：验证训练流程
- ✅ 解码器切换：验证不同解码器
- ✅ 环境变量：验证参数覆盖

## 🔄 与TranAD的集成

1. **保持vendor目录不变**：作为稳定参考
2. **自动检测vendor解码器**：优先使用原始实现
3. **统一接口设计**：保持与TranAD的兼容性
4. **环境变量控制**：灵活切换配置

## 📈 下一步计划

1. **Phase-2对抗目标**：精确迁入TranAD的对抗训练
2. **配置系统完善**：支持更多数据集和模型配置
3. **可视化增强**：AAC调度器曲线和分布漂移图表
4. **性能优化**：针对不同硬件的优化

## 🐛 故障排除

### 常见问题

1. **PyTorch未安装**
   ```bash
   conda install pytorch torchvision torchaudio -c pytorch
   ```

2. **依赖缺失**
   ```bash
   pip install scikit-learn
   ```

3. **权限问题**
   ```bash
   chmod +x scripts/*.sh
   ```

4. **路径问题**
   - 确保在项目根目录运行脚本
   - 检查Python路径设置

### 调试模式

```bash
# 启用详细输出
export ITAC_DEBUG=1
python test_itac_ad_quick.py
```

## 📝 开发指南

### 添加新解码器
1. 在 `tranad_decoders.py` 中添加新类
2. 在 `_try_vendor_decoders()` 中添加检测逻辑
3. 在 `ITAC_AD._build_decoders()` 中添加分支

### 添加新配置
1. 在 `configs/` 中添加YAML文件
2. 在脚本中加载配置
3. 更新环境变量映射

### 添加新测试
1. 在 `tests/` 中添加测试文件
2. 遵循 `test_*.py` 命名规范
3. 使用 `pytest` 运行

---

**🎉 恭喜！iTAC-AD 项目已准备就绪，可以开始您的异常检测研究之旅！**
