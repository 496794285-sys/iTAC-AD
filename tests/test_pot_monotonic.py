# -*- coding: utf-8 -*-
import numpy as np
from itac_eval.metrics import pot_threshold

def test_pot_monotonic():
    rng = np.random.default_rng(0)
    s = rng.normal(size=5000)
    t1 = pot_threshold(s, q=0.90, level=0.99)
    t2 = pot_threshold(s, q=0.95, level=0.99)
    assert t2 >= t1 - 1e-6  # q↑，阈值不应明显下降
