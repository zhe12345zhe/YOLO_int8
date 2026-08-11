"""INT8 训练引擎 (方案 B) 训练入口: 前向 int8 GEMM + SwitchBack dW, ultralytics 集成。

用法:
    python int8_engine_train.py --train --data datasets/coco-big/data.yaml --epochs 15
    python int8_engine_train.py --train --data /root/autodl-tmp/coco-full/data.yaml --epochs 15 --batch 32
    python int8_engine_train.py --bench --device 0        # 吞吐对比 (得先有权重加载)
    python int8_engine_train.py --sanity                   # 数值验证 (本机可跑)
训练产物: runs/detect/out/int8_engine_b/, 与 FP32/QAT 同目录树可对比。
"""
import argparse
import os
import time

import torch
from ultralytics import YOLO, SETTINGS

from int8_engine import patch_int8_engine, count_int8_convs, sanity_check

DEFAULT_DATA = "datasets/coco-big/data.yaml"


def patch_from_yolo(model):
    """对 ultralytics 模型挂 INT8 引擎 (trainer.model 或检测模型均可)。"""
    net = model.trainer.model if getattr(model, "trainer", None) else model.model
    # 加载的 ckpt 可能是推理态 (requires_grad=False), 训练前统一开启
    for p in net.parameters():
        p.requires_grad_(True)
    names = patch_int8_engine(net, verbose=True)
    return net, names


def run_train(args):
    model = YOLO(args.ckpt)

    def _on_start(trainer):
        for p in trainer.model.parameters():
            p.requires_grad_(True)
        patch_int8_engine(trainer.model, verbose=True)

    model.add_callback("on_train_start", _on_start)
    t0 = time.time()
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=8, seed=0, project="out", name=args.name,
                exist_ok=True, verbose=False, cache=False, amp=False,
                resume=args.resume)
    print(f"[int8_engine 训练] 用时 {time.time() - t0:.0f}s")


def run_bench(args):
    """fp16 训练步 vs int8 引擎训练步吞吐对比 (同图, 含 backward)。"""
    model = YOLO(args.ckpt)
    net0 = model.model
    dev = "cuda" if (args.device == "0" and torch.cuda.is_available()) else "cpu"
    x = torch.randn(args.batch, 3, args.imgsz, args.imgsz).to(dev)

    def step(net, iters=10):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(iters):
            y = net(x)
            loss = sum(v.sum() for v in y.values() if isinstance(v, torch.Tensor))
            loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return (time.perf_counter() - t0) / iters

    net1 = model.model.to(dev)
    t_fp16 = step(net1)
    patch_int8_engine(net1, verbose=True)
    t_int8 = step(net1)
    print(f"[bench] imgsz={args.imgsz} batch={args.batch} device={dev}")
    print(f"  fp16 训练:   {t_fp16 * 1e3:8.1f} ms/step  ({count_int8_convs(net1)} 层 int8)")
    print(f"  int8 训练:   {t_int8 * 1e3:8.1f} ms/step  加速 {t_fp16 / t_int8:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("train")
    p.add_argument("--ckpt", default="yolov8n.pt")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="int8_engine_b")
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("bench")
    p.add_argument("--ckpt", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")

    p = sub.add_parser("sanity")
    p.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    os.makedirs("out", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    SETTINGS.update(datasets_dir=os.path.abspath("datasets"))
    if args.cmd == "sanity":
        res = sanity_check(verbose=True)
        ok = res.get("gradcheck") and res.get("dw_rel_err", 1) < 0.10
        print(f"\n[{'PASS' if ok else 'FAIL'}] (backend={res['backend']})")
        return
    if args.cmd == "train":
        run_train(args)
    elif args.cmd == "bench":
        run_bench(args)


if __name__ == "__main__":
    main()