# -*- coding: utf-8 -*-
"""
JSON流脱敏工具
"""
SENSITIVE_KEYS = {"user", "email", "ip", "token", "id"}

def redact_inplace(d: dict, extra_keys=None):
    """就地脱敏字典中的敏感字段"""
    keys = set(SENSITIVE_KEYS) | set(extra_keys or [])
    for k in list(d.keys()):
        if k in keys and isinstance(d[k], (str, int, float)):
            d[k] = "***"
    return d
