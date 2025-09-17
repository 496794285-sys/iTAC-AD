# -*- coding: utf-8 -*-
import itertools, os, subprocess, json, numpy as np, pandas as pd

DATASET = os.environ.get("DATASET","SMD:machine-1-1")
SEED = os.environ.get("SEED","0")

grid = dict(
    AAC_TAU   =[0.85,0.90,0.95],
    AAC_ALPHA =[0.5,1.0,2.0],
    AAC_BETA  =[0.0,0.5,1.0],
    AAC_WND   =[128,256,512],
)

rows = []
for tau,alpha,beta,wnd in itertools.product(grid["AAC_TAU"], grid["AAC_ALPHA"], grid["AAC_BETA"], grid["AAC_WND"]):
    env = os.environ.copy()
    env.update({
        "AAC_TAU":str(tau), "AAC_ALPHA":str(alpha), "AAC_BETA":str(beta),
        "AAC_WND":str(wnd), "ITAC_NO_TB":"1","ITAC_NO_PLOTS":"1","ITAC_SAVE":"1","ITAC_LOG_EVERY":"50",
        "ITAC_FORCE_CPU": env.get("ITAC_FORCE_CPU","1")
    })
    print(f"[run] tau={tau} alpha={alpha} beta={beta} wnd={wnd}")
    subprocess.check_call(["python","main.py","--model","iTAC_AD","--dataset",DATASET,"--retrain"], env=env)
    run_dir = subprocess.check_output("ls -1dt outputs/* | head -n1", shell=True).decode().strip()
    subprocess.check_call(["bash","scripts/eval_event.sh", run_dir], env=env)
    m = json.load(open(os.path.join(run_dir,"metrics.json")))
    rows.append(dict(tau=tau,alpha=alpha,beta=beta,wnd=wnd, **m))

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/aac_sensitivity.csv", index=False)
print(df.sort_values("f1", ascending=False).head(10))
