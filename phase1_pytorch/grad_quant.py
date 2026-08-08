"""补验证: 反向传播的误差张量 E 是操作数吗? 把它 int8 化再对比。

数学定义 (train_compare.py 中已验证的等式):
    dw = A_q x E,  dx = W_q^T x E   —— E 是与 A_q/W_q 并列的乘法操作数。
目前 A_q/W_q 是 int8 网格值, 但 E 是 fp32。本脚本用 backward hook 把
每层的输入误差梯度 E 挂上同一个 int8 假量化 (quantize_int8),
即反传的 E_q 参与下游乘法:
    dw = A_q x E_q ;   dx = W_q^T x E_q
与"E 不动"相同任务/网络/seed 对比, 看精度差多少, 并输出 E 的
量化误差与 STE 恒等验证。

用法: python grad_quant.py [--epochs 15]
"""
import argparse
import time

import torch
import torch.nn as nn

import fake_quant as fq
from mini_yolo import MiniYOLO, detect_metrics, yolo_loss
from synthetic_data import get_loaders


def make_hooks(model, quantize):
    """给所有卷积挂"输入误差量化"hook; quantize=False 时是纯对比检查用。"""

    def hook(_m, gin, _gout):
        if not quantize:
            return gin
        out = []
        for g in gin:
            s = g.abs().max().clamp(min=1e-8) / 127.0
            out.append(fq.quantize_int8(g, s))
        return tuple(out)

    hs = []
    if quantize:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                hs.append(m.register_full_backward_hook(hook))
    return hs


def run(quantize_e, epochs, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    model = MiniYOLO(quantized=True)          # 与 Phase1 QAT 相同结构
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, val_loader = get_loaders(batch=32, train_n=3000, val_n=300)
    hooks = make_hooks(model, quantize_e)
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
    hit = evaluate(model, val_loader)
    print(f"  E-{'int8' if quantize_e else 'fp32'}: {epochs} epochs, 用时 {time.time() - t0:.0f}s, hit = {hit:.4f}")
    return hit


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    hits, n = 0.0, 0
    for x, t in loader:
        hits += detect_metrics(model(x), t) * len(t)
        n += len(t)
    return hits / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    print(f"反向传播操作数补验证: E 张量是否 int8 ({args.epochs} epochs)")
    ha = run(False, args.epochs)
    hb = run(True, args.epochs)
    print(f"\n结论: E-fp32 hit={ha:.4f} | E-int8 hit={hb:.4f} | 差 {hb - ha:+.4f}")


if __name__ == "__main__":
    main()