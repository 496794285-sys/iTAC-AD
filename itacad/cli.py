# -*- coding: utf-8 -*-
import argparse, os, sys, subprocess, json
from itacad.infer.predict import predict_csv
from itacad.export import export_onnx, export_torchscript

def main():
    ap = argparse.ArgumentParser(prog="itacad", description="iTAC-AD CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="batch predict on CSV")
    p_pred.add_argument("--csv", required=True)
    p_pred.add_argument("--ckpt", required=True)
    p_pred.add_argument("--window", type=int, required=True)
    p_pred.add_argument("--stride", type=int, default=1)
    p_pred.add_argument("--normalize", default="zscore", choices=["zscore","minmax","none"])
    p_pred.add_argument("--label-col", default=None)
    p_pred.add_argument("--out", default=None)
    p_pred.add_argument("--pot-q", type=float, default=0.98)
    p_pred.add_argument("--pot-level", type=float, default=0.99)
    p_pred.add_argument("--event-iou", type=float, default=0.1)
    p_pred.add_argument("--score-reduction", default="mean", choices=["mean","median","p95"])

    p_stream = sub.add_parser("stream", help="realtime stream from stdin")
    p_stream.add_argument("--ckpt", required=True)
    p_stream.add_argument("--L", type=int, required=True)
    p_stream.add_argument("--D", type=int, required=True)

    p_exp = sub.add_parser("export", help="export model")
    p_exp.add_argument("--ckpt", required=True)
    p_exp.add_argument("--format", choices=["ts","onnx"], default="onnx")
    p_exp.add_argument("--L", type=int, required=True)
    p_exp.add_argument("--D", type=int, required=True)
    p_exp.add_argument("--out", required=True)

    p_sjson = sub.add_parser("stream-json", help="realtime streaming from JSON/JSONL")
    p_sjson.add_argument("--ckpt", required=True)
    p_sjson.add_argument("--L", type=int, required=True)
    p_sjson.add_argument("--D", type=int, default=None)
    p_sjson.add_argument("--jsonl", default=None, help="JSONL file path; omit to read stdin")
    p_sjson.add_argument("--tail", action="store_true")
    p_sjson.add_argument("--poll", type=float, default=0.05)
    g = p_sjson.add_mutually_exclusive_group(required=True)
    g.add_argument("--vector-field")
    g.add_argument("--fields")
    g.add_argument("--prefix")
    p_sjson.add_argument("--pot-q", type=float, default=0.98)
    p_sjson.add_argument("--pot-level", type=float, default=0.99)

    args = ap.parse_args()
    if args.cmd == "predict":
        res = predict_csv(
            csv_path=args.csv, ckpt_dir=args.ckpt, window=args.window, stride=args.stride,
            normalize=args.normalize, pot_q=args.pot_q, pot_level=args.pot_level,
            event_iou=args.event_iou, label_col=args.label_col, out_dir=args.out,
            score_reduction=args.score_reduction
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "stream":
        # 将 stdin 行流转发给 stream_runner（避免重复依赖）
        code = subprocess.call([sys.executable, "rt/stream_runner.py", args.ckpt, str(args.L), str(args.D)])
        sys.exit(code)
    elif args.cmd == "export":
        if args.format == "ts":
            export_torchscript(args.ckpt, args.L, args.D, args.out)
        else:
            export_onnx(args.ckpt, args.L, args.D, args.out)
    elif args.cmd == "stream-json":
        cmd = [sys.executable, "rt/json_stream.py",
               "--ckpt", args.ckpt,
               "--L", str(args.L)]
        if args.D is not None: cmd += ["--D", str(args.D)]
        if args.jsonl: cmd += ["--jsonl", args.jsonl]
        if args.tail: cmd += ["--tail"]
        cmd += ["--poll", str(args.poll), "--pot-q", str(args.pot_q), "--pot-level", str(args.pot_level)]
        if args.vector_field:
            cmd += ["--vector-field", args.vector_field]
        elif args.fields:
            cmd += ["--fields", args.fields]
        else:
            cmd += ["--prefix", args.prefix]
        sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
