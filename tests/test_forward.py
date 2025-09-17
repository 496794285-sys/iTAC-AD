import torch
from itac_ad.models.itac_ad import ITAC_AD

def test_forward_shapes():
    B,T,D = 4, 96, 7
    x = torch.randn(B,T,D)
    m = ITAC_AD(feats=D)
    o1,o2,aux = m(x)
    assert o1.shape == (B,T,D)
    assert o2.shape == (B,T,D)
    assert "h_token" in aux and aux["h_token"].shape[:2] == (B,D)
