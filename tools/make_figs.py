# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/all.csv")
for ds, g in df.groupby("dataset"):
    fig = plt.figure()
    for tag, gg in g.groupby("tag"):
        xs = gg["seed"].values
        ys = gg["f1"].values
        plt.plot(xs, ys, marker="o", label=tag)
    plt.title(f"F1 vs seed - {ds}")
    plt.xlabel("seed"); plt.ylabel("F1")
    plt.legend()
    fig.savefig(f"results/fig_f1_seed_{ds.replace(':','_')}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
print("[make_figs] saved to results/")
