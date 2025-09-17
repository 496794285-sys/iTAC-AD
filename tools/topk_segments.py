# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def to_events(bits):
    bits = bits.astype(bool)
    ev = []
    i = 0
    n = len(bits)
    while i < n:
        if bits[i]:
            j = i
            while j + 1 < n and bits[j+1]: j += 1
            ev.append((i,j))
            i = j + 1
        else:
            i += 1
    return ev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--thr", type=float)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--missed-only", action="store_true")
    args = ap.parse_args()

    scores = np.load(os.path.join(args.run, "scores.npy"))
    labels = np.load(os.path.join(args.run, "labels.npy"))
    if args.thr is None:
        thr = float(np.quantile(scores, 0.99))
    else:
        thr = args.thr
    pred = (scores > thr).astype(int)
    ev_pred = to_events(pred)
    ev_true = to_events(labels)

    # 评分：片段最高分
    segs = []
    for s,e in ev_pred:
        mx = float(scores[s:e+1].max())
        hit = any(not (e2 < s or e < s2) for s2,e2 in ev_true)
        if args.missed_only and hit:
            continue
        segs.append(dict(start=s,end=e,hit=bool(hit),max_score=mx))
    segs = sorted(segs, key=lambda x: x["max_score"], reverse=True)[:args.k]
    df = pd.DataFrame(segs)
    os.makedirs(os.path.join(args.run,"diag"), exist_ok=True)
    out_csv = os.path.join(args.run,"diag","topk.csv")
    df.to_csv(out_csv, index=False)
    print(df)

    # 画图
    for i,seg in enumerate(segs):
        s,e = seg["start"], seg["end"]
        L = max(0, s-50); R = min(len(scores)-1, e+50)
        fig = plt.figure()
        plt.plot(np.arange(L,R+1), scores[L:R+1], label="score")
        plt.axhline(thr, linestyle="--", label="thr")
        plt.fill_between(np.arange(L,R+1), 0, 1, where=labels[L:R+1]>0, alpha=0.1, transform=plt.gca().get_xaxis_transform(), label="label")
        plt.title(f"seg{i}_hit={seg['hit']} [{s},{e}]")
        plt.legend()
        fig.savefig(os.path.join(args.run,"diag",f"seg_{i}_{s}_{e}.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
