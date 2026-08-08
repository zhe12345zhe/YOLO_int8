"""量化组合消融: W / A / E (误差) 三种操作数分别开/关量化, 统一训练对比。

用户需求: "权重先不需要量化, 把误差也量化再训练" -> 模式 (A+E int8, W fp32)。
本轮完整覆盖 5 种组合:

  T0 全 fp32                 (baseline)
  T1 W+A 量化, E 不量化      (现状 QAT)
  T2 W+A+E 全量化             (含误差操作数)
  T3 A+E 量化, W 不量化       (用户要求)
  T4 仅 E 量化                (误差单独量化)

用法: python train_modes.py [--epochs 15] [--seed 0]
"""
import argparse
import time

import torch
import torch.nn as nn

import fake_quant as fq
from mini_yolo import MiniYOLO, detect_metrics, yolo_loss
from synthetic_data import get_loaders


def e_hook(quant_e):
    def hook(_m, gin, _gout):
        if not quant_e:
            return gin
        return tuple(fq.quantize_int8(g, g.abs().max().clamp(min=1e-8) / 127.0) for g in gin)
    return hook


def train_run(quant_w, quant_a, quant_e, epochs, seed, lr=2e-3):
    torch.manual_seed(seed)
    model = MiniYOLO(quantized=True, quant_w=quant_w)
    if not quant_a:                       # 同结构 fp32 baseline 的对照 (仍保留量化层但关激活量化)
        for m in model.modules():
            if hasattr(m, "aq"):
                m.aq = None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = get_loaders(batch=32, train_n=3000, val_n=300)
    hooks = []
    if quant_e:
        hk = e_hook(True)
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                hooks.append(m.register_full_backward_hook(hk))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, t in train_loader:
            opt.zero_grad()
            loss = yolo_loss(model(x), t)
            loss.backward()
            opt.step()
    for h in hooks:
        h.remove()
    model.eval()
    hits, n = 0.0, 0
    with torch.no_grad():
        for x, t in val_loader:
            hits += detect_metrics(model(x), t) * len(t)
            n += len(t)
    return hits / n, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    modes = [
        ("fp32       (W,A,E)", False, False, False),
        ("QAT 现状    (W,A)", True,  True,   False),
        ("全量化  (W,A,E)", True,  True,   True),
        ("忽略W (A,E)",      False, True,   True),
        ("仅E (A,W否)",      False, False,  True),
    ]
    print(f"{'模式':22s} {'quant_W':8s} {'quant_A':8s} {'quant_E':8s} hit    用时")
    results = []
    for tag, w, a, e in modes:
        hit, dt = train_run(w, a, e, args.epochs, seed=0)
        results.append((tag, w, a, e, hit))
        print(f"{tag:22s} {str(w):8s} {str(a):8s} {str(e):8s} {hit:.4f}  {dt:.0f}s")
    print("\n对比 (用户要求的 A+E 组合 vs 现状):")
    for tag, _w, _a, _e, hit in results:
        print(f"  {tag:22s} hit = {hit:.4f}")


if __name__ == "__main__":
    main()