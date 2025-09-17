# -*- coding: utf-8 -*-
import os, torch
from typing import Optional
from itacad.infer.predict import load_model
from itacad.exportable import make_exportable

@torch.no_grad()
def _dummy_input(L:int, D:int, device:str):
    return torch.zeros(1, L, D, dtype=torch.float32, device=device)

def export_torchscript(ckpt_dir: str, L: int, D: int, out_path: str):
    model, device, _ = load_model(ckpt_dir)
    
    # 强制使用CPU进行导出，避免设备不匹配问题
    model = model.to('cpu')
    device = 'cpu'
    
    # 强制构建模型
    dummy = _dummy_input(L, D, device)
    with torch.no_grad():
        _ = model(dummy, phase=1, aac_w=0.0)
    
    exp = make_exportable(model).to(device).eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 先尝试 script（更严格但图更干净），失败再 trace
    try:
        scripted = torch.jit.script(exp)
        print("[export] Using TorchScript script method")
    except Exception as e:
        print(f"[export] Script failed: {e}, falling back to trace")
        scripted = torch.jit.trace(exp, (dummy,), strict=False)
    scripted = torch.jit.freeze(scripted)
    scripted.save(out_path)
    print(f"[export] TorchScript saved -> {out_path}")

def export_onnx(ckpt_dir: str, L: int, D: int, out_path: str):
    model, device, _ = load_model(ckpt_dir)
    
    # 强制使用CPU进行导出，避免设备不匹配问题
    model = model.to('cpu')
    device = 'cpu'
    
    # 强制构建模型
    dummy = _dummy_input(L, D, device)
    with torch.no_grad():
        _ = model(dummy, phase=1, aac_w=0.0)
    
    exp = make_exportable(model).to(device).eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 优先使用 PyTorch 2.x 的 dynamo_export（更强的 Python 捕获能力）
    tried_dynamo = False
    try:
        from torch.onnx import dynamo_export
        exported = dynamo_export(exp, dummy)
        exported.save(out_path)  # 直接保存为 ONNX
        print(f"[export] ONNX (dynamo_export) saved -> {out_path}")
        return
    except Exception as e:
        tried_dynamo = True
        print(f"[export] dynamo_export failed: {e} (will fallback)")

    # 回退到传统导出（需要静态友好的前向；我们的 wrapper 满足）
    try:
        torch.onnx.export(
            exp, (dummy,), out_path,
            input_names=["W"], output_names=["O1","score"],
            dynamic_axes={
                "W": {0: "batch", 1: "time"},
                "O1": {0: "batch", 1: "time"},
                "score": {0: "batch"},
            },
            opset_version=17
        )
        print(f"[export] ONNX (classic) saved -> {out_path}")
    except Exception as e2:
        how = "dynamo_export & classic" if tried_dynamo else "classic"
        raise RuntimeError(f"ONNX export failed via {how}: {e2}") from e2