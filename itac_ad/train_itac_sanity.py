import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from itac_ad.models.itac_ad import ITAC_AD
from itac_ad.components.aac_scheduler import AACScheduler

def make_synthetic(batch=32, T=96, D=7, steps=200):
    # 产生一个带异常的多变量序列：正弦 + 噪声 + 罕见尖脉冲
    X = []
    Y = []
    for _ in range(steps):
        t = torch.linspace(0, 10, T)
        base = torch.stack([torch.sin(t * (i+1)) for i in range(D)], dim=-1)
        noise = 0.05 * torch.randn(T, D)
        x = base + noise
        # 随机插入异常
        if torch.rand(1).item() < 0.3:
            idx_t = torch.randint(0, T, (1,)).item()
            idx_d = torch.randint(0, D, (1,)).item()
            x[idx_t, idx_d] += torch.randn(1).item() * 5.0
        X.append(x)
        Y.append(base)   # 目标=无噪/无异常的基线（仅用于训练演示）
    X = torch.stack(X).float()      # [steps, T, D]
    Y = torch.stack(Y).float()
    # mini-batches
    for i in range(0, steps, batch):
        yield X[i:i+batch], Y[i:i+batch]

def main():
    device = "cpu"
    model = ITAC_AD(d_model=128, n_heads=8, e_layers=2, dropout=0.1).to(device)
    aac = AACScheduler(window_size=256, quantile_p=0.9)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    l1 = nn.L1Loss()
    print(">> sanity train start")
    it = 0
    for epoch in range(2):
        for xb, yb in make_synthetic():
            xb, yb = xb.to(device), yb.to(device)
            o1, o2, _ = model(xb)

            # 残差与自适应 w_t
            resid = (o1 - yb).abs()
            w_t = aac.step(resid)

            # Phase-1: 重构贴近 (L1)
            loss_rec = l1(o1, yb)
            # Phase-2: 对抗项（把 o2 拉离 yb 的目标；用 -L1 作为“反向”目标）
            loss_adv = -l1(o2, yb)

            loss = loss_rec + w_t * loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 20 == 0:
                print(f"[epoch {epoch}] it={it:04d} loss={loss.item():.4f} w={w_t:.3f} (rec={loss_rec.item():.4f}, adv={loss_adv.item():.4f})")
            it += 1
    print("<< sanity train done")

if __name__ == "__main__":
    main()
