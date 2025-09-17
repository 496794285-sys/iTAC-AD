# -*- coding: utf-8 -*-
# iTAC-AD stable runner (algorithm-equivalent outer loop, macOS-safe)
from __future__ import annotations
import os, sys, gc, csv, math, time
from datetime import datetime
from pathlib import Path
import argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====== 项目根路径（动态获取）======
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ====== 导入你的算法组件（不改算法本体）======
from itac_ad.models.itac_ad import ITAC_AD
from itac_ad.components.aac_scheduler import AACScheduler
from itacad.thresholding import fit_threshold, save_threshold
from itacad.score import score_windows, make_windows

# ====== 运行控制（只改"怎么跑"，不改"算什么"）======
ITAC_FORCE_CPU   = os.getenv("ITAC_FORCE_CPU","1") not in ("0","false","False")
ITAC_EPOCHS_ENV  = os.getenv("ITAC_EPOCHS")
ITAC_MAX_STEPS   = int(os.getenv("ITAC_MAX_STEPS","0"))       # 每个 epoch 最多步（0=不限）
ITAC_LOG_EVERY   = int(os.getenv("ITAC_LOG_EVERY","1"))
ITAC_SAVE        = os.getenv("ITAC_SAVE","1") not in ("0","false","False")
ITAC_SKIP_EVAL   = os.getenv("ITAC_SKIP_EVAL","0") not in ("0","false","False")
ITAC_NUM_WORKERS = int(os.getenv("ITAC_NUM_WORKERS","0"))     # macOS 建议 0
ITAC_PIN_MEMORY  = os.getenv("ITAC_PIN_MEMORY","0") not in ("0","false","False")
ITAC_PERSISTENT  = os.getenv("ITAC_PERSISTENT","0") not in ("0","false","False")
LR               = float(os.getenv("ITAC_LR","1e-3"))

# 不引入任何交互式绘图或 TB 线程；只做 CSV 与 checkpoint 输出

def set_seed(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

def pick_device():
    if ITAC_FORCE_CPU: return torch.device("cpu")
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
        return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

# ====== 极小"合成数据"烟测集（只替代数据 IO，算法不变）======
# 你的算法在 synthetic 上的损失定义/前向保持不变；这里仅把数据喂给模型。
class TinySineDataset(Dataset):
    """返回 (x,y) 形状 [T,D]，DataLoader 会批成 [B,T,D]。"""
    def __init__(self, N=32, T=96, D=7, noise=0.05):
        self.N, self.T, self.D, self.noise = N, T, D, noise
        t = torch.linspace(0, 2*math.pi, T)
        base = torch.stack([torch.sin(t*(i+1)) for i in range(D)], dim=-1)  # [T,D]
        self.base = base.float()
    def __len__(self): return self.N
    def __getitem__(self, idx):
        y = self.base
        x = y + self.noise * torch.randn_like(y)
        return x, y

def make_loader(ds, batch_size=8, shuffle=True):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=ITAC_NUM_WORKERS,
        pin_memory=ITAC_PIN_MEMORY if ITAC_NUM_WORKERS>0 else False,
        persistent_workers=ITAC_PERSISTENT if ITAC_NUM_WORKERS>0 else False,
    )

def csv_logger(path: Path):
    f = path.open("w", newline="")
    w = csv.DictWriter(f, fieldnames=["epoch","step","loss","loss_rec","loss_adv","w"])
    w.writeheader()
    return f, w

# ====== 训练与评测（算法等价：Phase-1/Phase-2 + AAC + GRL）======
def train_one_epoch(model, opt, aac, loader, device, epoch:int=0, max_steps:int=0, log_every:int=1):
    model.train()
    steps = 0; last_print = time.time()
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device); yb = yb.to(device)        # [B,T,D]
        
        # 计算残差用于AAC（使用目标yb而不是输入xb）
        resid = (model(xb, phase=1)["O1"] - yb).abs()  # Phase-1重构误差
        w_t = float(aac.step(resid))                  # ← AAC 保持
        
        # Phase-2 训练：使用AAC权重
        result = model(xb, phase=2, aac_w=w_t)
        o1, o2 = result["O1"], result["O2"]
        loss_rec, loss_adv = result["loss_rec"], result["loss_adv"]
        loss = result["loss"]  # 已经包含了aac_w权重

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        steps += 1
        if (i % max(1, log_every)) == 0 or (time.time()-last_print) > 10:
            print(f"[train] epoch={epoch} step={i} loss={loss.item():.6f} rec={loss_rec.item():.6f} adv={loss_adv.item():.6f} w={w_t:.3f}", flush=True)
            last_print = time.time()

        if max_steps and steps >= max_steps:
            break
    # 明确释放，防 resource_tracker
    del loader; gc.collect()
    return steps, float(loss.item()), float(loss_rec.item()), float(loss_adv.item()), w_t

@torch.no_grad()
def evaluate(model, loader, device, max_steps:int=0, out_dir: Path = None):
    if loader is None: return 0
    model.eval(); steps = 0
    all_scores = []
    all_labels = []
    
    for i, (xb, yb) in enumerate(loader):
        result = model(xb.to(device), phase=2, aac_w=0.0)
        # 计算异常分数：重构误差的L2范数
        scores = torch.norm(result["O1"] - xb, dim=-1).cpu().numpy()  # [B, T]
        all_scores.append(scores.flatten())
        # 对于synthetic数据，我们使用简单的阈值来生成标签
        labels = (scores > scores.mean() + 2 * scores.std()).astype(int)
        all_labels.append(labels.flatten())
        steps += 1
        if max_steps and steps >= max_steps: break
    
    # 保存评测结果
    if out_dir and len(all_scores) > 0:
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        np.save(out_dir / "scores.npy", scores)
        np.save(out_dir / "labels.npy", labels)
        print(f"[eval] saved scores.npy ({len(scores)} points) and labels.npy ({labels.sum()} anomalies)")
    
    del loader; gc.collect()
    return steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset","-d", default="synthetic", choices=["synthetic"])
    parser.add_argument("--model","-m", default="iTAC_AD")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    set_seed(42)
    device = pick_device(); print(f"[device] {device}", flush=True)

    # 数据（与你的算法解耦；算法本体保持一致）
    if args.dataset == "synthetic":
        train_ds = TinySineDataset(N=64, T=96, D=7, noise=0.05)
        test_ds  = TinySineDataset(N=16, T=96, D=7, noise=0.05)
    else:
        raise ValueError("只提供 synthetic 烟测；真实集成见脚注说明")

    train_loader = make_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = make_loader(test_ds,  batch_size=args.batch_size, shuffle=False) if not ITAC_SKIP_EVAL else None

    # 模型 + AAC + 优化器（算法不变）
    model = ITAC_AD(feats=7).to(device)  # synthetic数据有7个特征
    
    # 触发模型构建（lazy_build需要前向传播）
    dummy_input = torch.randn(1, 96, 7).to(device)  # [B,T,D]
    with torch.no_grad():
        _ = model(dummy_input)
    
    aac = AACScheduler()                     # 支持你已加过的 env 覆盖参数
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    epochs = int(ITAC_EPOCHS_ENV) if ITAC_EPOCHS_ENV else args.epochs

    # 输出目录（与原外壳一致）
    out_dir = ROOT / "vendor" / "tranad" / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S-itac")
    out_dir.mkdir(parents=True, exist_ok=True)
    fcsv, wcsv = csv_logger(out_dir / "train.csv")

    # 训练 + 可选评测（保证能退出）
    total_steps = 0
    for ep in range(epochs):
        steps, loss, lrec, ladv, w_t = train_one_epoch(
            model, opt, aac, train_loader, device,
            epoch=ep, max_steps=ITAC_MAX_STEPS, log_every=ITAC_LOG_EVERY
        )
        total_steps += steps
        # 记录最后一步
        wcsv.writerow({"epoch": ep, "step": steps, "loss": loss, "loss_rec": lrec, "loss_adv": ladv, "w": w_t})
        fcsv.flush()

        if not ITAC_SKIP_EVAL:
            _ = evaluate(model, test_loader, device, max_steps=0, out_dir=out_dir)

    # 保存（可控）
    if ITAC_SAVE:
        ckpt = out_dir / "itac_ad_ckpt.pt"
        torch.save({"model": model.state_dict()}, ckpt)
        print(f"[save] {ckpt}", flush=True)
        
        # 训练完成后：阈值拟合
        print("[calibrate] 开始阈值拟合...", flush=True)
        
        # 1) 收集训练集数据并计算统计量
        train_data = []
        for xb, yb in train_loader:
            train_data.append(xb.numpy())
        X_train = np.concatenate(train_data, axis=0)  # [N,T,D]
        mu = X_train.mean(axis=(0,1))  # [D]
        sigma = X_train.std(axis=(0,1)) + 1e-8  # [D]
        
        # 保存训练集统计量
        import json
        train_stats = {"mean": mu.tolist(), "std": sigma.tolist()}
        with open(out_dir / "train_stats.json", "w") as f:
            json.dump(train_stats, f, indent=2)
        print(f"[calibrate] 保存训练统计量: mean={mu.mean():.4f}, std={sigma.mean():.4f}")
        
        # 2) 训练集残差得分（滑窗 -> 前向 -> 评分）
        L = int(os.getenv("ITAC_WINDOW", "96"))  # 窗口长度
        W = make_windows(X_train.reshape(-1, X_train.shape[-1]), L=L, stride=1)  # [N,L,D]
        
        # 使用新的评分函数
        agg_method = os.getenv("ITAC_SCORE_AGG", "topk_mean")
        topk_ratio = float(os.getenv("ITAC_TOPK", "0.1"))
        
        S_train = score_windows(
            model, W, mu, sigma,
            agg=agg_method,
            topk_ratio=topk_ratio,
            device=device
        )
        np.save(out_dir / "train_scores.npy", S_train)
        print(f"[calibrate] 训练集分数: min={S_train.min():.4f}, max={S_train.max():.4f}, mean={S_train.mean():.4f}")
        
        # 3) 阈值拟合（POT，失败回退）+ 落盘
        pot_q = float(os.getenv("POT_Q", "0.98"))
        pot_level = float(os.getenv("POT_LVL", "0.99"))
        
        thr, meta = fit_threshold(
            S_train,
            q=pot_q, 
            level=pot_level,
            method="auto",  # 默认 Pickands，失败回退 quantile
            max_tail=5000, 
            top_clip=0.999, 
            seed=0, 
            k=2000
        )
        
        # 添加额外的元数据
        meta.update({
            "agg": agg_method,
            "topk_ratio": topk_ratio,
            "train_scores_stats": {
                "min": float(S_train.min()),
                "max": float(S_train.max()),
                "mean": float(S_train.mean()),
                "std": float(S_train.std())
            }
        })
        
        save_threshold(str(out_dir), thr, meta)
        print(f"[calibrate] 阈值拟合完成: threshold={thr:.4f}, method={meta.get('method', 'unknown')}")

    try: fcsv.close()
    except: pass
    gc.collect()
    print(f"[done] clean exit after {total_steps} steps; logs -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()