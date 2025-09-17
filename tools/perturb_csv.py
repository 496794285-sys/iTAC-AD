# -*- coding: utf-8 -*-
import argparse, numpy as np, pandas as pd, os

def add_missing(X, rate, mode="bernoulli"):
    X = X.copy()
    if rate <= 0: return X
    n, d = X.shape
    m = np.random.rand(n, d) < rate
    X[m] = np.nan
    # 简单填充：列均值
    col_mean = np.nanmean(X, axis=0, keepdims=True)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1], axis=1)
    return X

def add_spikes(X, rate, scale=3.0):
    X = X.copy()
    if rate <= 0: return X
    n, d = X.shape
    k = int(n * rate)
    idx = np.random.choice(n, size=max(1,k), replace=False)
    noise = np.random.randn(len(idx), d) * scale
    X[idx] = X[idx] + noise
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--missing", type=float, default=0.0, help="缺失率 [0,1]")
    ap.add_argument("--spike", type=float, default=0.0, help="随机尖峰率（按行）[0,1]")
    ap.add_argument("--spike-scale", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    labels = None
    if args.label_col and args.label_col in df.columns:
        labels = df[args.label_col].values
        X = df.drop(columns=[args.label_col]).values.astype("float32")
    else:
        X = df.values.astype("float32")

    X = add_missing(X, args.missing)
    X = add_spikes(X, args.spike, args.spike_scale)

    out_df = pd.DataFrame(X, columns=[c for c in df.columns if c != args.label_col])
    if labels is not None:
        out_df[args.label_col] = labels
        cols = [c for c in df.columns if c != args.label_col] + [args.label_col]
        out_df = out_df[cols]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[perturb] wrote -> {args.out}")

if __name__ == "__main__":
    main()