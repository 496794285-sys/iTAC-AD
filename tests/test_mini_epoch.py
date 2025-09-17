import torch, torch.nn as nn, torch.optim as optim
from itac_ad.models.itac_ad import ITAC_AD
from itac_ad.components.aac_scheduler import AACScheduler

def _data(steps=20, B=8, T=64, D=5):
    for _ in range(steps):
        t = torch.linspace(0, 6.28, T)
        base = torch.stack([torch.sin(t*(i+1)) for i in range(D)], dim=-1)
        x = base + 0.05*torch.randn(T,D)
        yield x[None].repeat(B,1,1), base[None].repeat(B,1,1)

def test_mini_epoch():
    m = ITAC_AD(feats=5)
    aac = AACScheduler()
    data_iter = _data()
    first_batch = next(data_iter)
    with torch.no_grad():
        m(first_batch[0])
    opt = optim.Adam(m.parameters(), lr=1e-3)
    l1 = nn.L1Loss()
    from itertools import chain
    for i, (xb, yb) in enumerate(chain((first_batch,), data_iter)):
        o1,o2,_ = m(xb)
        loss = l1(o1,yb) + aac.step((o1-yb).abs()) * (-l1(o2,yb))
        opt.zero_grad(); loss.backward(); opt.step()
        if i==3: break  # 迷你训练几步即通过
