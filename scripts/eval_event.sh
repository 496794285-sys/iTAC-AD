#!/usr/bin/env bash
set -e
RUN_DIR="${RUN_DIR:-${1:-outputs/latest}}"
POT_Q="${POT_Q:-0.98}"
POT_LEVEL="${POT_LEVEL:-0.99}"
EVENT_IOU="${EVENT_IOU:-0.1}"

export RUN_DIR="$RUN_DIR"
export POT_Q="$POT_Q"
export POT_LEVEL="$POT_LEVEL"
export EVENT_IOU="$EVENT_IOU"

python - <<'PY'
import json, os, numpy as np
from itac_eval.metrics import pot_threshold, binarize_by_threshold, event_f1, pr_auc, point_f1

run = os.environ.get("RUN_DIR", "outputs/latest")
pot_q = float(os.environ.get("POT_Q","0.98"))
pot_level = float(os.environ.get("POT_LEVEL","0.99"))
event_iou = float(os.environ.get("EVENT_IOU","0.1"))

print(f"[eval] using run_dir: {run}")
scores = np.load(os.path.join(run, "scores.npy"))
labels = np.load(os.path.join(run, "labels.npy"))

thr = pot_threshold(scores, q=pot_q, level=pot_level)
pred = binarize_by_threshold(scores, thr)

# 计算事件级和点级指标
ev = event_f1(pred, labels, iou_thresh=event_iou)
pt = point_f1(pred, labels)
auc_pr = pr_auc(scores, labels)

metrics = dict(threshold=thr, auc_pr=auc_pr, **ev)
metrics.update(pt)
print("[eval]", json.dumps(metrics, ensure_ascii=False, indent=2))

with open(os.path.join(run, "metrics.json"), "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
PY
