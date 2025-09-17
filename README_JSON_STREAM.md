# iTAC-AD JSON流处理功能使用指南

## 概述

iTAC-AD现在支持实时JSON流处理，可以直接接入日志/消息队列/传感器总线等数据源。支持三种数据提取方式，自动维度推断，以及在线异常检测。

## 功能特性

- ✅ **三种数据提取方式**：vector-field、fields、prefix
- ✅ **自动维度推断**：无需预先指定特征维度
- ✅ **在线阈值估计**：使用POT算法动态调整异常阈值
- ✅ **多种输入源**：支持stdin和文件输入
- ✅ **文件尾追踪**：支持tail -f风格的实时监控
- ✅ **统一JSON输出**：便于集成到告警系统

## 命令格式

```bash
itacad stream-json --ckpt <模型路径> --L <窗口大小> [选项]
```

### 必需参数

- `--ckpt`: 模型检查点目录路径
- `--L`: 滑动窗口大小（必须大于等于模型训练时的窗口大小）

### 数据提取方式（三选一）

- `--vector-field <路径>`: 指定JSON中向量字段的路径（支持点路径如 `data.values`）
- `--fields <字段列表>`: 指定多个标量字段，用逗号分隔（如 `temp,press,flow`）
- `--prefix <前缀>`: 自动收集带指定前缀的数值字段（如 `f_` 会收集 `f_0`, `f_1` 等）

### 可选参数

- `--D <维度>`: 手动指定特征维度（默认自动推断）
- `--jsonl <文件路径>`: JSONL文件路径（默认从stdin读取）
- `--tail`: 启用文件尾追踪模式（类似tail -f）
- `--poll <秒数>`: 文件尾追踪的轮询间隔（默认0.05秒）
- `--pot-q <分位数>`: POT算法的分位数参数（默认0.98）
- `--pot-level <置信度>`: POT算法的置信度参数（默认0.99）

## 使用示例

### 1. 从stdin读取JSON（向量在values字段）

```bash
# 生成测试数据
python - <<'PY' | itacad stream-json --ckpt outputs/your_ckpt --L 100 --vector-field values
import json, sys, time, numpy as np
np.random.seed(0)
for t in range(1200):
    v = np.random.randn(25).tolist()
    if 500<=t<=540: v = (np.array(v)+4.0).tolist()   # 注入异常
    print(json.dumps({"ts": t, "values": v})); sys.stdout.flush(); time.sleep(0.005)
PY
```

### 2. 从文件读取（多个标量字段）

```bash
# 假设 log.jsonl 每行形如 {"ts":..., "temp":1.2, "press":3.4, "flow":5.6}
itacad stream-json --ckpt outputs/your_ckpt --L 100 \
  --jsonl log.jsonl --fields temp,press,flow
```

### 3. 使用前缀自动收集特征

```bash
# 键名如 f_0..f_37，会按键名排序拼成38维向量
itacad stream-json --ckpt outputs/your_ckpt --L 128 \
  --jsonl sensors.jsonl --prefix f_
```

### 4. 实时文件监控（tail -f风格）

```bash
# 监控实时日志文件
itacad stream-json --ckpt outputs/your_ckpt --L 50 \
  --jsonl /var/log/sensors.jsonl --tail --vector-field data.values
```

## 输出格式

程序输出统一的JSON事件流，每行一个JSON对象：

### 系统就绪事件
```json
{"event": "ready", "L": 100, "D": 25, "source": "stdin"}
```

### 维度推断事件
```json
{"event": "infer_dim", "D": 25}
```

### 处理事件
```json
{"event": "tick", "score": 0.123, "thr": 0.456, "anom": 1, "ts": 1234}
```

### 错误事件
```json
{"event": "error", "msg": "dim_mismatch got=24 expect=25"}
```

### 跳过事件
```json
{"event": "skip", "reason": "no_vector"}
```

## 字段说明

- `event`: 事件类型（ready/infer_dim/tick/error/skip）
- `score`: 重构误差分数
- `thr`: 当前异常阈值
- `anom`: 异常标志（0=正常，1=异常）
- `ts`: 时间戳（如果输入数据包含）
- `msg`: 错误消息（仅错误事件）

## 集成示例

### 与Kafka集成
```bash
# 从Kafka消费数据并处理
kafka-console-consumer --topic sensors | \
itacad stream-json --ckpt models/itac_ad --L 64 --vector-field values | \
jq 'select(.event=="tick" and .anom==1)' | \
# 发送到告警系统
curl -X POST http://alerts/api/anomaly -d @-
```

### 与日志系统集成
```bash
# 监控应用日志
tail -f /var/log/app.log | \
grep "metrics" | \
itacad stream-json --ckpt models/itac_ad --L 32 --fields cpu,mem,disk | \
# 过滤异常并记录
jq 'select(.anom==1)' >> /var/log/anomalies.log
```

## 性能优化建议

1. **窗口大小**: 根据数据特性和延迟要求选择合适的窗口大小
2. **轮询间隔**: 文件尾追踪时，根据数据更新频率调整poll参数
3. **阈值参数**: 根据业务需求调整POT算法的q和level参数
4. **批处理**: 对于高吞吐场景，考虑批量处理多个样本

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径和权限
2. **维度不匹配**: 确保输入数据维度与训练时一致
3. **JSON解析错误**: 检查输入数据格式
4. **内存不足**: 减少窗口大小或批处理大小

### 调试模式

```bash
# 启用详细输出
itacad stream-json --ckpt models/itac_ad --L 10 --vector-field values 2>&1 | tee debug.log
```

## 测试验证

运行完整测试套件：
```bash
python test_complete_json_stream.py
```

这将测试所有数据提取方式和功能特性。
