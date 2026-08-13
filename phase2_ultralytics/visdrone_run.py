"""VisDrone 微调: FP32 / QAT(W+A) / QAT(W+A+E) 三方案 (基于 qat_run_big, 独立目录名)。"""
import argparse
import os
import time

import torch
from ultralytics import YOLO, SETTINGS

from qat_patch import patch_qat

DATA = "/root/autodl-tmp/VisDrone/VisDrone_Dataset/visdrone.yaml"
STAGES = {"fp32": ("vis_fp32", None),
          "wa": ("vis_wa", False),
          "wae": ("vis_wae", True)}


def find_ckpt(name, suffix="best.pt"):
    cands = [
        os.path.join("runs", "detect", "out", name, "weights", suffix),
        os.path.join("out", name, "weights", suffix),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def run_one(name, quant_e, tag, args):
    best_ckpt = find_ckpt(name, "best.pt")
    if best_ckpt:
        model = YOLO(best_ckpt, task="detect")
        print(f"[{tag}] 已有权重, 直接评估: {best_ckpt}")
    else:
        model = YOLO("yolov8n.pt")
    if quant_e is not None:
        model.add_callback("on_train_start",
                           lambda trainer: patch_qat(trainer.model, quant_act=True,
                                                     quant_e=quant_e, verbose=True))
    t0 = time.time()
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device="0", workers=8, seed=0, project="out", name=name,
                exist_ok=True, verbose=False, cache=False, amp=False)
    print(f"[{tag} 微调] 用时 {time.time() - t0:.0f}s")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stage", type=str, default="all",
                    choices=["all", "fp32", "wa", "wae"])
    args = ap.parse_args()

    torch.set_num_threads(max(4, torch.get_num_threads()))
    os.makedirs("out", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    SETTINGS.update(datasets_dir=os.path.abspath("datasets"))

    order = ["fp32", "wa", "wae"] if args.stage == "all" else [args.stage]
    for s in order:
        name, qe = STAGES[s]
        run_one(name, qe, s.upper(), args)
        print(f"[{s}] 完成")


if __name__ == "__main__":
    main()
