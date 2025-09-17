# -*- coding: utf-8 -*-
import os, sys, json, time
from collections import deque
from typing import List, Optional
import numpy as np
import torch

# 复用你已有的加载与在线阈值（POT）逻辑
from itacad.infer.predict import load_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rt.stream_runner import OnlinePOT  # 我们之前写过的在线POT类
from utils.redact import redact_inplace  # 脱敏工具

def _get_by_path(obj, path: str):
    """支持 a.b.c 的点路径取值。"""
    cur = obj
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def _extract_vector(
    data: dict,
    *,
    vector_field: Optional[str],
    fields: Optional[List[str]],
    prefix: Optional[str]
) -> Optional[List[float]]:
    if vector_field:
        v = _get_by_path(data, vector_field)
        if isinstance(v, (list, tuple)) and all(isinstance(x,(int,float)) for x in v):
            return [float(x) for x in v]
        return None
    if fields:
        out = []
        for k in fields:
            v = _get_by_path(data, k)
            if v is None or not isinstance(v, (int, float)):
                return None
            out.append(float(v))
        return out
    if prefix:
        cand = []
        for k, v in data.items():
            if isinstance(v, (int, float)) and str(k).startswith(prefix):
                cand.append((k, float(v)))
        if not cand:
            return None
        # 以键名的自然排序保证稳定拼接
        cand.sort(key=lambda kv: kv[0])
        return [x for _, x in cand]
    return None

def _iter_lines(jsonl_path: Optional[str], tail: bool, poll: float):
    """从 stdin 或文件(支持tail -f) 持续读行。"""
    if not jsonl_path:
        for line in sys.stdin:
            yield line
        return
    # 文件模式
    with open(jsonl_path, "r") as f:
        if tail:
            # 跳到末尾（只看新追加）
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll)
                    continue
                yield line
        else:
            # 静态文件模式：读取所有行后退出
            for line in f:
                yield line

def stream_json(
    ckpt_dir: str,
    L: int,
    D: Optional[int],
    jsonl_path: Optional[str],
    *,
    vector_field: Optional[str],
    fields: Optional[List[str]],
    prefix: Optional[str],
    q: float = 0.98,
    level: float = 0.99,
    poll: float = 0.05,
    tail: bool = False
):
    model, device, _ = load_model(ckpt_dir, device="cpu")
    buf = deque(maxlen=L)
    thr_est = OnlinePOT(q=q, level=level, wnd=4096, refit=256)

    print(json.dumps({"event":"ready","L":L,"D":D or "auto","source": jsonl_path or "stdin"}), flush=True)

    for raw in _iter_lines(jsonl_path, tail, poll):
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            # 脱敏处理
            redact_keys = os.environ.get("ITAC_REDACT_KEYS", "").split(",")
            redact_keys = [k.strip() for k in redact_keys if k.strip()]
            redact_inplace(data, extra_keys=redact_keys)
        except Exception as e:
            print(json.dumps({"event":"error","msg":f"bad_json: {e}"}), flush=True)
            continue

        vec = _extract_vector(data, vector_field=vector_field, fields=fields, prefix=prefix)
        if vec is None:
            print(json.dumps({"event":"skip","reason":"no_vector"}), flush=True)
            continue

        # 维度推断/校验
        d_now = len(vec)
        if D is None:
            D = d_now
            print(json.dumps({"event":"infer_dim","D":D}), flush=True)
        if d_now != D:
            print(json.dumps({"event":"error","msg":f"dim_mismatch got={d_now} expect={D}"}), flush=True)
            continue

        # 背压保护：写慢时丢弃旧样本
        if len(buf) == buf.maxlen:
            buf.popleft()  # 丢最旧，软背压
        buf.append(vec)
        if len(buf) < L:
            continue

        w = torch.tensor([list(buf)], dtype=torch.float32, device=device)  # (1,L,D)
        with torch.no_grad():
            # 只用 Phase-1 重构
            if "phase" in model.forward.__code__.co_varnames:
                out = model(w, phase=1, aac_w=0.0)
                O1 = out.get("O1", None)
                if O1 is None: O1 = out  # 兼容实现
            else:
                O1 = model(w)
            rec = (O1 - w).abs().mean().item()

        thr = thr_est.update(rec)
        is_anom = int(rec > thr)
        data_out = {
            "event":"tick",
            "score": rec,
            "thr": thr,
            "anom": is_anom
        }
        # 透传原始时间戳/ID 如果有
        for k in ("ts","timestamp","id"):
            if k in data: data_out[k] = data[k]
        print(json.dumps(data_out), flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="iTAC-AD JSONL streaming")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--jsonl", default=None, help="JSONL file path; omit for stdin")
    ap.add_argument("--tail", action="store_true", help="follow file (tail -f)")
    ap.add_argument("--poll", type=float, default=0.05, help="tail poll seconds")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vector-field", help="dot.path to list vector field")
    g.add_argument("--fields", help="comma-separated scalar keys, in order")
    g.add_argument("--prefix", help="pick numeric keys starting with prefix, sorted")

    ap.add_argument("--pot-q", type=float, default=0.98)
    ap.add_argument("--pot-level", type=float, default=0.99)
    args = ap.parse_args()

    fields = args.fields.split(",") if args.fields else None
    stream_json(
        args.ckpt, args.L, args.D, args.jsonl,
        vector_field=args.vector_field, fields=fields, prefix=args.prefix,
        q=args.pot_q, level=args.pot_level, poll=args.poll, tail=args.tail
    )
