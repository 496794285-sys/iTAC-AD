# -*- coding: utf-8 -*-
import numpy as np
from itac_eval.metrics import event_f1, binarize_by_threshold

def test_event_merge_basic():
    # 1,1,0,1,1 -> 两个事件
    pred = np.array([1,1,0,1,1,0,0])
    true = np.array([1,1,0,1,1,0,0])
    m = event_f1(pred, true, iou_thresh=0.1)
    assert m["tp"]==2 and m["fp"]==0 and m["fn"]==0 and abs(m["f1"]-1.0)<1e-6
