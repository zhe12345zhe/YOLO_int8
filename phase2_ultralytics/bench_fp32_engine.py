"""FP32 手搓引擎 vs int8 引擎: 训练 + 推理速度 (同架构对比, 3080Ti, bs16 imgsz320)。"""
import os, sys, time
sys.path.insert(0, "/root/dl/proj/phase2_ultralytics")
os.environ["PATH"] = "/root/miniconda3/bin:" + os.environ.get("PATH", "")
import torch
from ultralytics import YOLO, SETTINGS
import int8_engine as ie
from fp32_engine import patch_fp32_engine

SETTINGS.update(datasets_dir="/root/dl/proj/phase2_ultralytics/datasets")
os.makedirs("/root/dl/proj/phase2_ultralytics/datasets", exist_ok=True)
dev = "cuda"
torch.backends.cudnn.benchmark = True

def build(mode):
    m = YOLO("yolov8n.pt")
    n = m.model.to(dev)
    for p in n.parameters():
        p.requires_grad_(True)
    n.train()
    if mode == "fp32_engine":
        patch_fp32_engine(n, verbose=False)
    elif mode == "int8_engine":
        ie.patch_int8_engine(n, verbose=False)
    return n

def bench(fn, iters=30, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

x = torch.randn(16, 3, 320, 320, device=dev)

print("=== 训练速度 (fwd+bwd, bs16 imgsz320) ===")
for mode in ("fp32_engine", "int8_engine"):
    n = build(mode)
    def step():
        y = n(x)
        if isinstance(y, dict):
            tot = sum(v.sum() for v in y.values() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point)
        else:
            tot = y.sum()
        tot.backward()
    t = bench(step)
    print(f"  {mode:12s}: {t:8.2f} ms/step  ({1000/t:6.2f} it/s)")
    del n
    torch.cuda.empty_cache()

print("=== 推理速度 (eval 前向, bs16 imgsz320) ===")
for mode in ("fp32_engine", "int8_engine"):
    m = YOLO("yolov8n.pt")
    n = m.model.to(dev).eval()
    for p in n.parameters():
        p.requires_grad_(False)
    if mode == "fp32_engine":
        patch_fp32_engine(n, verbose=False)
        n.eval()
    elif mode == "int8_engine":
        ie.patch_int8_engine(n, verbose=False)
        n.eval()
    def fwd():
        with torch.no_grad():
            y = n(x)
        return y
    t = bench(fwd, iters=100, warm=30)
    print(f"  {mode:12s}: {t:8.2f} ms/step")
    del n
    torch.cuda.empty_cache()
