# -*- coding: utf-8 -*-
import pandas as pd, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="results/all.csv")
args = ap.parse_args()

df = pd.read_csv(args.csv)
# 取 seed 均值±std
agg = df.groupby(["tag","dataset"]).agg(
    f1_mean=("f1","mean"), f1_std=("f1","std"),
    pr=("precision","mean"), rc=("recall","mean"), auc_pr=("auc_pr","mean")
).reset_index()

# 输出 Markdown 表
print("\n### Aggregated Results\n")
wide = agg.pivot(index="dataset", columns="tag", values="f1_mean")
print(wide.round(3))

agg.to_csv("results/agg_table.csv", index=False)
