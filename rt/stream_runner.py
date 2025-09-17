# -*- coding: utf-8 -*-
import sys, os, yaml, json, time, numpy as np, torch
from collections import deque
from itac_eval.metrics import pot_threshold  # 用作初始阈值；在线时转为高分位自适应
from itacad.infer.predict import load_model

class OnlinePOT:
    """简化在线阈值：维护分位数 + 尾部缓冲，定期重估。"""
    def __init__(self, q=0.98, level=0.99, wnd=4096, refit=512):
        self.q, self.level = q, level
        self.buf = deque(maxlen=wnd)
        self.refit = refit
        self._thr = None
        self._steps = 0

    def update(self, s: float) -> float:
        self.buf.append(float(s)); self._steps += 1
        if self._thr is None or self._steps % self.refit == 0:
            arr = np.array(self.buf, dtype=float)
            if len(arr)>64:
                self._thr = pot_threshold(arr, q=self.q, level=self.level)
            else:
                self._thr = np.quantile(arr, self.level if len(arr)>0 else 1.0)
        return self._thr

def stream_from_stdin(ckpt_dir: str, L: int, D: int, q=0.98, level=0.99):
    model, device, cfg = load_model(ckpt_dir, device="cpu", feats=D)  # 强制使用CPU
    buf = deque(maxlen=L)
    thr_est = OnlinePOT(q=q, level=level, wnd=4096, refit=256)

    print(json.dumps({"event":"ready","L":L,"D":D}), flush=True)
    for line in sys.stdin:
        line=line.strip()
        if not line: 
            continue
        # 期望格式：用逗号分隔的 D 个数
        vals = [float(x) for x in line.split(",")]
        if len(vals) != D:
            print(json.dumps({"event":"error","msg":f"dim!={D}"}), flush=True); 
            continue
        buf.append(vals)
        if len(buf) < L: 
            continue
        w = torch.tensor([list(buf)], dtype=torch.float32, device="cpu")  # 强制使用CPU
        with torch.no_grad():
            out = model(w, phase=1, aac_w=0.0) if "phase" in model.forward.__code__.co_varnames else model(w)
            O1 = out["O1"] if isinstance(out, dict) and "O1" in out else out
            rec = (O1 - w).abs().mean().item()
        thr = thr_est.update(rec)
        is_anom = int(rec > thr)
        print(json.dumps({"event":"tick","score":rec,"thr":thr,"anom":is_anom}), flush=True)

if __name__=="__main__":
    # 示例： python rt/stream_runner.py outputs/ckpt_dir 100 38
    ckpt_dir = sys.argv[1]; L = int(sys.argv[2]); D = int(sys.argv[3])
    stream_from_stdin(ckpt_dir, L, D)
