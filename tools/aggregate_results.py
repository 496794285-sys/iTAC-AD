# -*- coding: utf-8 -*-
import argparse, os, json, pandas as pd, hashlib, subprocess

def git_hash():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", default="results/all.csv")
    args = ap.parse_args()

    mfile = os.path.join(args.run, "metrics.json")
    cfgfile = os.path.join(args.run, "config.yaml")
    metrics = json.load(open(mfile)) if os.path.exists(mfile) else {}
    row = dict(
        tag=args.tag, dataset=args.dataset, seed=int(args.seed),
        git=git_hash(), run=args.run,
        threshold=metrics.get("threshold"), f1=metrics.get("f1"),
        precision=metrics.get("precision"), recall=metrics.get("recall"),
        auc_pr=metrics.get("auc_pr")
    )
    df = pd.DataFrame([row])
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["tag","dataset","seed","run"], keep="last")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.tail(1))

if __name__ == "__main__":
    main()
