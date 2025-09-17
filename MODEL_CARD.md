# Model Card — iTAC-AD

## Intended Use
Multivariate time-series anomaly detection for industrial telemetry / IT ops.

## Data
SMD, SMAP, MSL, PSM（请附来源链接/许可说明）。声明：不包含个人敏感信息。

## Training
Backbone: iTransformer (variate tokens). Adversarial Phase-2 with GRL.
AACScheduler adjusts adversarial weight by residual quantiles & distribution drift.

## Metrics
Event-level F1 (IoU=0.1), PR AUC, POT (q=0.98, level=0.99). See `results/`.

## Limitations
- 对突发分布外异常（极少见模式）可能过度保守；
- 阈值依赖数据域分布；跨域迁移需重新校准；
- 实时流对乱序/缺帧敏感，请使用乱序缓冲与缺失填补。

## Safety / Privacy
- 支持在线推理中对敏感字段脱敏与不落盘（参见 `utils/redact.py`）。
- 不应在包含个人身份信息(PII)的原始日志上长期存储未脱敏数据。
