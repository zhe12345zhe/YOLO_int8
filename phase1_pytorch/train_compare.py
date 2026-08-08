"""Phase 1 主脚本: 训练对比 fp32 / QAT / TrueGrad(对照) / PTQ, 并验证反向传播操作数量化。

用法:
    python train_compare.py --quick          # 快速跑 (每阶段 15/15/4 epoch)
    python train_compare.py                   # 完整 (每阶段 40/40/8 epoch)

输出 (out/ 目录):
    1. 损失与 hit-rate 曲线 (CSV + 打印表格)
    2. 操作数验证报告:
       a. STE 恒等: 量化前输入 x 的梯度 == 量化后输入 x_q 的梯度 (手动卷积转置核对)
       b. 权重梯度操作数: autograd 的 w.grad == unfold(x_q) x E  (x_q 是 int8 激活)
       c. 权重 int8 量化误差与各层激活 scale
"""
import argparse
import csv
import time

import torch
import torch.nn.functional as F

from fake_quant import ActQuant, WeightQuant
from mini_yolo import MiniYOLO, detect_metrics, yolo_loss
from quant_conv import QuantConv2d
from synthetic_data import get_loaders

DEVICE = torch.device("cpu")
OUT = "out"


def set_ste(model, ste):
    for m in model.modules():
        if isinstance(m, (ActQuant, WeightQuant)):
            m.ste = ste


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tl, hits, n = 0.0, 0.0, 0
    for x, t in loader:
        pred = model(x)
        tl += yolo_loss(pred, t).item() * len(t)
        hits += detect_metrics(pred, t) * len(t)
        n += len(t)
    return tl / n, hits / n


def train(model, loader, epochs, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for x, t in loader:
            opt.zero_grad()
            pred = model(x)
            loss = yolo_loss(pred, t)
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            vl, vr = evaluate(model, val_loader)
            print(f"  ep {ep:3d} | train_loss {tl / len(loader):.4f} | val_loss {vl:.4f} | hit@{'%.2f' % 0.5} {vr:.3f}")
    return tl / len(loader)


def make_ptq_model(src: MiniYOLO) -> MiniYOLO:
    """PTQ: 用 fp32 训练好的权重构造量化网络, 不再训练 (只 eval)。"""
    dst = MiniYOLO(in_ch=src.c1.in_channels, quantized=True)
    names = ["c1", "c2", "c3", "head_f"]
    for n in names:
        dst.get_submodule(n).conv.weight.data.copy_(src.get_submodule(n).weight.data)
        dst.get_submodule(n).conv.bias.data.copy_(src.get_submodule(n).bias.data)
    dst.head_box.weight.data.copy_(src.head_box.weight.data)
    dst.head_box.bias.data.copy_(src.head_box.bias.data)
    dst.head_obj.weight.data.copy_(src.head_obj.weight.data)
    dst.head_obj.bias.data.copy_(src.head_obj.bias.data)
    return dst


def verify_operands(model, batch, layer_name="c1"):
    """验证反向传播的操作数确实是量化值 (STE 与 x_q 操作数两条证据)。"""
    x, _ = batch
    x = x.detach().requires_grad_(True)
    layer = model.get_submodule(layer_name)

    y1 = layer(x)                       # y1 = conv(x_q, w_q)
    loss = y1.sum()
    loss.backward()

    w_q = layer.saved_w_q
    x_q = layer.saved_x_q
    E = torch.ones_like(y1)             # dL/dY (局部验证用全 1 足够)

    # (a) STE: autograd 的 x.grad 应等于用量化权重反卷积得到的梯度
    g_x_auto = x.grad
    g_x_manual = F.conv_transpose2d(E, w_q, stride=layer.conv.stride,
                                    padding=layer.conv.padding)
    err_ste = (g_x_auto - g_x_manual).abs().max().item()
    rel_ste = err_ste / (g_x_manual.abs().max().item() + 1e-8)

    # (b) 权重梯度操作数: autograd 的 w.grad == unfold(x_q) x E
    w_grad_auto = layer.conv.weight.grad
    patches = F.unfold(x_q, kernel_size=3, padding=1)      # B, C_in*9, L
    L = patches.shape[-1]
    E_flat = E.reshape(E.shape[0], E.shape[1], L)
    w_grad_manual = torch.einsum("bcl,bkl->ck", E_flat, patches).reshape(w_grad_auto.shape)
    err_op = (w_grad_auto - w_grad_manual).abs().max().item()
    rel_op = err_op / (w_grad_manual.abs().max().item() + 1e-8)

    # (c) 权重量化误差与 scale
    w = layer.conv.weight
    wq = layer.wq.dequant_weight(w)
    werr = (w - wq).abs().max().item() / w.abs().max().item()
    return rel_ste, rel_op, werr


def run(mode, epochs, loader, seed=0):
    torch.manual_seed(seed)
    model = MiniYOLO(quantized=True if mode != "fp32" else False)
    print(f"[{mode.upper()}] {epochs} epochs")
    t0 = time.time()
    if mode == "fp32":
        train(model, loader, epochs)
        torch.save(model.state_dict(), f"{OUT}/fp32.pt")
        vl, vr = evaluate(model, val_loader)
    elif mode == "qat":
        train(model, loader, epochs)
        torch.save(model.state_dict(), f"{OUT}/qat.pt")
        vl, vr = evaluate(model, val_loader)
    elif mode == "true":                     # 真实导数对照: 预期训练停滞
        set_ste(model, ste=False)
        train(model, loader, epochs)
        vl, vr = evaluate(model, val_loader)
    elif mode == "ptq":                      # 直接评估 fp32 权重的量化版本
        src = MiniYOLO(quantized=False)
        src.load_state_dict(torch.load(f"{OUT}/fp32.pt"))
        model = make_ptq_model(src)
        vl, vr = evaluate(model, val_loader)
    print(f"  done in {time.time() - t0:.1f}s | val_loss {vl:.4f} | hit@{0.5} {vr:.3f}")
    return model, vl, vr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    import os
    os.makedirs(OUT, exist_ok=True)

    train_loader, val_loader = get_loaders()

    E = dict(fp32=40, qat=40, true=8)
    if args.quick:
        E = dict(fp32=15, qat=15, true=4)

    results = {}
    models = {}
    for mode in ["fp32", "qat", "true", "ptq"]:
        models[mode], vl, vr = run(mode, E.get(mode, 0), train_loader)
        results[mode] = (vl, vr)

    print("\n========== 对比结果 (val) ==========")
    for m, (vl, vr) in results.items():
        print(f"  {m:6s}: val_loss {vl:.4f} | hit@0.5 {vr:.3f}")

    print("\n========== 反向传播操作数验证 (QAT 模型, c1 层) ==========")
    x, _ = next(iter(val_loader))
    rel_ste, rel_op, werr = verify_operands(models["qat"], (x, _))
    print(f"  (a) STE 恒等:  x 梯度 vs convT(E, W_q)  相对误差 = {rel_ste:.2e}  (≈0 => 梯度直通量化算子)")
    print(f"  (b) 权重梯度操作数: w.grad vs unfold(A_q)xE  相对误差 = {rel_op:.2e}  (≈0 => 操作数确为 int8 激活 A_q)")
    print(f"  (c) 权重 int8 量化误差 (max|W - W_q|/max|W|) = {werr:.2%}")
    print("\n  各层激活 scale (A_q 的 int8 粒度):")
    for name, m in models["qat"].named_modules():
        if isinstance(m, QuantConv2d) and m.aq is not None:
            print(f"    {name:10s} act_scale = {m.aq.last_scale.item():.4e}   w_scale(mean) = {m.wq.last_scale.mean().item():.4e}")

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "val_loss", "hit@0.5"])
        for m, (vl, vr) in results.items():
            w.writerow([m, f"{vl:.4f}", f"{vr:.3f}"])
    print(f"\n结果已写入 {OUT}/results.csv")
