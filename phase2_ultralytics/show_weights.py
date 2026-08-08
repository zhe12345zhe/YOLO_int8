"""打印 QAT 权重 vs FP32 权重 (第 1/12/24/40 层卷积), 并展示其 int8 量化网格效果。

用法: python show_weights.py [--qat <best.pt>] [--fp32 <best.pt>]
注意: 兼容 Windows 控制台, 输出统一写 UTF-8 (不带 BOM) 文件 weights_dump.txt。
"""
import argparse
import io
import sys

import torch

QMIN, QMAX = -128, 127

_OUT = io.StringIO()  # 收集所有输出, 最后统一用 UTF-8 写盘


def say(*a):
    print(*a)
    print(*a, file=_OUT)


def show(name, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ck["model"]
    convs = [x for x in m.modules() if isinstance(x, torch.nn.Conv2d)]
    say(f"\n{'=' * 62}\n[{name}] {ckpt_path}\n  共 {len(convs)} 个 Conv2d")
    for i in (0, 5, 11, 23, 39, len(convs) - 1):
        if i >= len(convs):
            continue
        c = convs[i]
        w = c.weight.detach()
        s = w.abs().amax(dim=(1, 2, 3)) / QMAX          # per-channel scale (同 qat_patch)
        wq = torch.clamp(torch.round(w / s.view(-1, 1, 1, 1)), QMIN, QMAX)  # int8 量化值
        err = (w - wq * s.view(-1, 1, 1, 1)).abs().max().item() / w.abs().max().item()
        flat = w.flatten()
        say(f"\n  layer[{i}]: weight {tuple(w.shape)}  (前 6 个浮点权重) {['%.4f' % v for v in flat[:6].tolist()]}")
        say(f"    int8 量化后: {wq.flatten()[:6].tolist()}")
        say(f"    范围 [{flat.min().item():.4f}, {flat.max().item():.4f}]  |  量化误差 {err:.4%}")
    return m


ap = argparse.ArgumentParser()
ap.add_argument("--qat", default=r"C:\Users\lenovo\runs\detect\out\qat\weights\best.pt")
ap.add_argument("--fp32", default=r"C:\Users\lenovo\runs\detect\out\fp32\weights\best.pt")
args = ap.parse_args()
show("QAT-50ep", args.qat)
show("FP32-50ep", args.fp32)
with open("weights_dump.txt", "w", encoding="utf-8-sig") as f:   # UTF-8 带 BOM, VS Code 可直接识别
    f.write(_OUT.getvalue())
print("\n[ok] 完整输出已写入 weights_dump.txt (UTF-8 with BOM)")