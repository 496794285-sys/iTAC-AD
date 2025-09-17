# -*- coding: utf-8 -*-
import os, pytest, torch, numpy as np

@pytest.mark.skipif(os.environ.get("SKIP_GRL_TEST","0")=="1", reason="skipped by env")
def test_grl_direction():
    # 需要 iTAC_AD 模型可导入，且 forward 支持 phase=2 和返回 O1/O2
    try:
        from itac_ad.models.itac_ad import ITAC_AD  # 你的模型路径
    except ImportError:
        pytest.skip("iTAC_AD model not available")
    
    torch.manual_seed(0)
    B,L,D = 8, 32, 5
    model = ITAC_AD()  # 假定用默认构造；若需要，可改成从 config 读取
    enc = model.encoder
    for p in model.dec1.parameters(): p.requires_grad=False  # 只看 enc/dec2
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    W = torch.randn(B, L, D)
    with torch.no_grad():
        out = model(W, phase=2, aac_w=1.0)
        o1_0 = (out["O1"]-W).abs().mean().item()
        o2_0 = (out["O2"]-W).abs().mean().item()

    for _ in range(5):
        out = model(W, phase=2, aac_w=1.0)
        loss = out["loss"]
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        out = model(W, phase=2, aac_w=1.0)
        o1_1 = (out["O1"]-W).abs().mean().item()
        o2_1 = (out["O2"]-W).abs().mean().item()

    # 期望：O2差距被推大（GRL生效），O1差距被拉小
    assert o1_1 <= o1_0 + 1e-4
    assert o2_1 >= o2_0 - 1e-4
