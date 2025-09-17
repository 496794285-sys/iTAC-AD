# -*- coding: utf-8 -*-
import argparse, pandas as pd

# TranAD paper (VLDB'22) Table 2, full-data results (SMAP/MSL/SMD)
TRANAD_PAPER = {
    "SMAP": {"precision": 0.8043, "recall": 1.0,    "f1": 0.8915, "auc_roc": 0.9921},
    "MSL":  {"precision": 0.9038, "recall": 1.0,    "f1": 0.9494, "auc_roc": 0.9916},
    "SMD":  {"precision": 0.9262, "recall": 0.9974, "f1": 0.9605},  # AUC(ROC)未强制对比
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all.csv")
    ap.add_argument("--out_csv", default="results/itac_vs_tranad_paper.csv")
    ap.add_argument("--out_md",  default="results/itac_vs_tranad_paper.md")
    ap.add_argument("--model_name", default="iTAC_AD")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["model"] == args.model_name].copy()
    # 多次seed则聚合
    keep_cols = [c for c in ["dataset","precision","recall","f1","auc_pr","auc_roc","seed"] if c in df.columns]
    df = df[keep_cols]
    # 兼容数据集别名：SMD:machine-1-1 -> SMD
    df["dataset"] = df["dataset"].str.replace(r"^SMD:.*$", "SMD", regex=True)
    g = df.groupby("dataset").agg({c: "mean" for c in df.columns if c not in ["dataset"]}).reset_index()

    rows = []
    for ds in ["SMAP","MSL","SMD"]:
        if ds not in g["dataset"].values: 
            continue
        row = g[g["dataset"]==ds].iloc[0].to_dict()
        itac_p, itac_r, itac_f1 = row.get("precision", float("nan")), row.get("recall", float("nan")), row.get("f1", float("nan"))
        ref = TRANAD_PAPER[ds]
        ref_p, ref_r, ref_f1 = ref["precision"], ref["recall"], ref["f1"]
        delta = itac_f1 - ref_f1
        rel = (delta / ref_f1) * 100.0
        rows.append({
            "dataset": ds,
            "iTAC_P": itac_p, "iTAC_R": itac_r, "iTAC_F1": itac_f1,
            "TranAD_Paper_P": ref_p, "TranAD_Paper_R": ref_r, "TranAD_Paper_F1": ref_f1,
            "ΔF1(iTAC-TranAD)": delta, "RelΔF1(%)": rel,
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    # Markdown 便于贴 README
    def fmt(x): 
        return "-" if pd.isna(x) else f"{x:.4f}"
    md = ["| dataset | iTAC-P | iTAC-R | iTAC-F1 | TranAD-P | TranAD-R | TranAD-F1 | ΔF1 | RelΔF1(%) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in out.iterrows():
        md.append("| {ds} | {p1} | {r1} | {f1} | {p2} | {r2} | {f2} | {d} | {rel} |".format(
            ds=r["dataset"],
            p1=fmt(r["iTAC_P"]), r1=fmt(r["iTAC_R"]), f1=fmt(r["iTAC_F1"]),
            p2=fmt(r["TranAD_Paper_P"]), r2=fmt(r["TranAD_Paper_R"]), f2=fmt(r["TranAD_Paper_F1"]),
            d=fmt(r["ΔF1(iTAC-TranAD)"]), rel=fmt(r["RelΔF1(%)"])
        ))
    with open(args.out_md,"w") as f: f.write("\n".join(md))
    print(f"[ok] wrote {args.out_csv} and {args.out_md}")

if __name__ == "__main__":
    main()