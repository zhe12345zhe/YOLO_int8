"""Phase 2 扩展实验: 权重 W + 激活 A + 误差梯度 E 三个操作数全部 int8, COCO128。"""

import argparse
import os
import time

import torch
from ultralytics import YOLO, SETTINGS

from qat_patch import patch_qat, count_quantized, calibrate_activations
from qat_run import make_calib_loader, calibrate, eval_map

# 已跑完的对照数字 (W+A 两步, 同数据/seed): 见 run_coco_full.txt
BASELINE = {"FP32": (0.6835, 0.5290), "PTQ_W+A": (0.6766, 0.5089), "QAT_W+A": (0.6969, 0.5281)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="coco128")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    torch.set_num_threads(max(4, torch.get_num_threads()))
    os.makedirs("out", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    SETTINGS.update(datasets_dir=os.path.abspath("datasets"))

    qat = YOLO("yolov8n.pt")
    qat.add_callback("on_train_start",
                     lambda trainer: patch_qat(trainer.model, quant_act=True, quant_e=True, verbose=True))
    t0 = time.time()
    qat.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              device="cpu", workers=0, seed=0, project="out", name="qat_e",
              exist_ok=True, verbose=False, cache=False, amp=False)
    print(f"[QAT(W+A+E) 微调] {args.epochs} epochs 用时 {time.time() - t0:.0f}s")

    net = qat.trainer.model
    nw, na, ne = count_quantized(net)
    print(f"[QAT(W+A+E)] 训练用网络量化卷积 {nw} 个 (激活 {na} + 误差E {ne})")
    qat.model = net
    calibrate(qat.model, args.data, args.imgsz)
    m50, map50_95 = eval_map(qat, args.data, imgsz=args.imgsz)
    print(f"[QAT(W+A+E) 微调后] mAP50 = {m50:.4f}  mAP50-95 = {map50_95:.4f}")

    fp, pe = BASELINE["QAT_W+A"]
    print("\n========== COCO128 全操作数量化对比 ==========")
    print(f"  方案                mAP50       mAP50-95")
    print(f"  FP32(上界)         {BASELINE['FP32'][0]:.4f}      {BASELINE['FP32'][1]:.4f}")
    print(f"  PTQ(W+A)           {BASELINE['PTQ_W+A'][0]:.4f}      {BASELINE['PTQ_W+A'][1]:.4f}")
    print(f"  QAT W+A            {fp:.4f}      {pe:.4f}")
    print(f"  QAT W+A+E(本次)    {m50:.4f}      {map50_95:.4f}   (vs W+A: {m50 - fp:+.4f} / {map50_95 - pe:+.4f})")
    with open("out/phase2_e_results.txt", "w", encoding="utf-8") as f:
        f.write(f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}\n")
        f.write(f"FP32 {BASELINE['FP32'][0]:.4f} {BASELINE['FP32'][1]:.4f}\n")
        f.write(f"QAT_W+A {fp:.4f} {pe:.4f}\n")
        f.write(f"QAT_W+A+E {m50:.4f} {map50_95:.4f}\n")


if __name__ == "__main__":
    main()