"""训练峰值显存对比: 原生 FP32 vs FP32 手搓 vs int8 引擎 (bs16 imgsz320, 3080Ti)。"""
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

x = torch.randn(16, 3, 320, 320, device=dev)

print("=== 训练峰值显存 (fwd+bwd, bs16 imgsz320) ===")
for mode in ("native", "fp32_engine", "int8_engine"):
    if mode == "native":
        m = YOLO("yolov8n.pt")
        n = m.model.to(dev)
        for p in n.parameters():
            p.requires_grad_(True)
        n.train()
    else:
        n = build(mode)
    # 预热 + 重置峰值统计
    y = n(x)
    if isinstance(y, dict):
        tot = sum(v.sum() for v in y.values() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point)
    else:
        tot = y.sum()
    tot.backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    y = n(x)
    if isinstance(y, dict):
        tot = sum(v.sum() for v in y.values() if isinstance(v, torch.Tensor) and v.dtype.is_floating_point)
    else:
        tot = y.sum()
    tot.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    cur = torch.cuda.memory_allocated() / 2**30
    print(f"  {mode:12s}: 峰值 {peak:6.2f} GB | 当前 {cur:6.2f} GB")
    del n
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
