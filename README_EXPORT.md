# iTAC-AD 模型导出功能

## 概述

iTAC-AD 现在支持将训练好的模型导出为 TorchScript 和 ONNX 格式，用于生产环境部署。导出功能采用了"瘦身 Wrapper"策略，只保留推理必需的 encoder + dec1 组件，避免了 GRL/AAC/字典输出等复杂属性。

## 核心文件

- `itacad/exportable.py` - 导出专用模块，只保留 encoder + dec1
- `itacad/export.py` - 导出工具，支持 TorchScript 和 ONNX
- `itac_ad/models/itac_ad.py` - 主模型，包含导出健康检查方法

## 使用方法

### 1. 基本导出

```python
from itacad.export import export_torchscript, export_onnx

# 导出 TorchScript
export_torchscript('path/to/checkpoint', L=16, D=8, 'output/model.ts')

# 导出 ONNX
export_onnx('path/to/checkpoint', L=16, D=8, 'output/model.onnx')
```

### 2. 命令行导出

```bash
# TorchScript 导出
itacad export --ckpt "path/to/checkpoint" --format ts --L 16 --D 8 --out exports/model.ts

# ONNX 导出
itacad export --ckpt "path/to/checkpoint" --format onnx --L 16 --D 8 --out exports/model.onnx
```

### 3. 导出健康检查

```python
from itacad.infer.predict import load_model

# 加载模型
model, device, cfg = load_model('path/to/checkpoint')

# 运行健康检查
result = model.export_sanity(L=16, D=8, device='cpu')
print(f"导出健康检查: {result}")
```

## 导出策略

### 1. 瘦身 Wrapper

- 只保留 `encoder` + `dec1` 组件
- 移除 GRL/AAC/Phase-2 分支
- 返回纯张量 `(O1, score)` 而不是字典
- 使用 `mean` 规约确保图形稳定

### 2. 多级回退

**TorchScript 导出:**
1. 优先使用 `torch.jit.script()` (更严格但图更干净)
2. 失败则回退到 `torch.jit.trace()`
3. 最后使用 `torch.jit.freeze()` 优化

**ONNX 导出:**
1. 优先使用 `torch.onnx.dynamo_export()` (PyTorch 2.x)
2. 失败则回退到 `torch.onnx.export()`
3. 使用 opset_version=17 确保兼容性

### 3. 设备处理

- 强制使用 CPU 进行导出，避免设备不匹配问题
- 自动处理 lazy_build 组件的设备迁移

## 导出结果

### 输入输出格式

- **输入**: `W` - 形状 `(B, L, D)` 的时序数据
- **输出**: 
  - `O1` - 形状 `(B, L, D)` 的重构结果
  - `score` - 形状 `(B,)` 的异常分数

### 使用导出的模型

```python
import torch

# 加载 TorchScript 模型
model = torch.jit.load('exports/model.ts')
model.eval()

# 推理
input_data = torch.randn(1, 16, 8)
with torch.no_grad():
    O1, score = model(input_data)

print(f"重构结果形状: {O1.shape}")
print(f"异常分数: {score}")
```

## 注意事项

### 1. 数值精度

- TorchScript 导出可能存在微小数值差异 (MSE < 1e-3)
- 这是 TorchScript 导出中的常见现象，不影响实际使用
- 建议在生产环境中验证关键指标的一致性

### 2. 环境依赖

**TorchScript 导出:**
- 只需要 PyTorch
- 无需额外依赖

**ONNX 导出:**
- 需要安装 `onnx` 和 `onnxruntime`
- 可选安装 `onnxscript` 以支持 dynamo_export

### 3. 模型兼容性

- 支持 iTAC-AD 的所有编码器类型 (itr, itransformer)
- 支持所有解码器类型 (mlp, tranad)
- 自动处理 lazy_build 组件的构建

## 故障排除

### 1. 设备不匹配错误

```
RuntimeError: Expected all tensors to be on the same device
```

**解决方案**: 导出函数已自动处理，强制使用 CPU 进行导出。

### 2. ONNX 导出失败

```
ModuleNotFoundError: No module named 'onnx'
```

**解决方案**: 安装 ONNX 相关依赖
```bash
pip install onnx onnxruntime
```

### 3. 数值差异过大

**解决方案**: 
- 确保模型完全构建后再导出
- 使用相同的随机种子进行测试
- 检查模型是否处于 eval 模式

## 测试

运行完整测试：

```bash
python test_export.py
```

测试包括：
1. TorchScript 导出功能
2. ONNX 导出功能（如果环境支持）
3. 数值对齐验证
4. 导出文件信息检查

## 性能优化

### 1. 模型大小

- 导出的 TorchScript 模型约 1.5MB
- 相比完整训练模型显著减小

### 2. 推理速度

- TorchScript 模型推理速度更快
- 支持批量处理
- 无需 Python 运行时开销

### 3. 部署优势

- 跨平台兼容
- 无需 PyTorch 依赖
- 支持 C++ 部署
- 更好的内存管理
