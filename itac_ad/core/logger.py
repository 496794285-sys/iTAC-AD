# itac_ad/core/logger.py
from __future__ import annotations
import csv, os, time
from typing import Dict

class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = None
    def log(self, row: Dict):
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row); self._file.flush()
    def close(self):
        self._file.close()

class TbLogger:
    def __init__(self, logdir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            self.w = None; return
        os.makedirs(logdir, exist_ok=True)
        self.w = SummaryWriter(logdir)
        self.step = 0
    def log_scalars(self, scalars: Dict):
        if not self.w: return
        for k,v in scalars.items():
            self.w.add_scalar(k, v, self.step)
        self.step += 1
    def close(self):
        if self.w: self.w.close()
