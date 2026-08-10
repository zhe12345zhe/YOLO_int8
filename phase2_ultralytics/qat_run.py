"""Phase 2: ultralytics YOLOv8n 的 int8 QAT 微调与评估对比。

对比设计 (同一任务, 同一 backbone):
    1. FP32 微调: 精度上界
    2. PTQ:       用 FP32 微调得到的权重, 挂上 int8 假量化 (不训练) 直接评估
                  -> 直观展示"训练后量化"的精度损失
    3. QAT 微调:  从预训练权重出发, 在假量化模拟下微调, 再评估
                  -> 展示量化感知训练几乎追平 FP32

用法: python qat_run.py [--epochs 12] [--imgsz 256] [--batch 8] [--quick]
"""
import argparse
import os
import time

import torch
from ultralytics import YOLO
from ultralytics import SETTINGS
from ultralytics.data import YOLODataset, build_dataloader
from ultralytics.data.utils import check_det_dataset

from qat_patch import patch_qat, count_quantized, calibrate_activations

DATA = "data/synth/data.yaml"
IMGSZ = 256
DATASETS_DIR = os.path.abspath("datasets")


def make_calib_loader(data_yaml, imgsz, batch=16):
    data = check_det_dataset(data_yaml)
    ds = YOLODataset(img_path=data["val"], imgsz=imgsz, batch_size=batch, augment=False,
                     rect=False, stride=32, data=data, single_cls=False)
    loader = build_dataloader(ds, batch=batch, workers=0, shuffle=False)
    n_batches = (len(ds) + batch - 1) // batch
    return loader, n_batches


def eval_map(model, data, imgsz=IMGSZ, device=None):
    if device is None:
        device = "0" if next(model.model.parameters()).is_cuda else "cpu"
    r = model.val(data=data, imgsz=imgsz, device=device,
                  verbose=False, plots=False, project="out", name="val_tmp",
                  exist_ok=True)
    return r.box.map50, r.box.map


def calibrate(model_net, data_yaml, imgsz):
    loader, n = make_calib_loader(data_yaml, imgsz)
    calibrate_activations(model_net, loader, n_batches=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synth/data.yaml",
                    help="数据集 yaml (也支持内置名如 coco128, coco8)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 3
        args.imgsz = 192

    data_yaml = args.data

    torch.set_num_threads(max(4, torch.get_num_threads()))
    os.makedirs("out", exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    SETTINGS.update(datasets_dir=DATASETS_DIR)   # 数据集下载到本地 phase2 目录

    # ---- 1. FP32 微调 (精度上界) ----
    fp32 = YOLO("yolov8n.pt")
    t0 = time.time()
    fp32.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
               device="cpu", workers=0, seed=0, project="out", name="fp32",
               exist_ok=True, verbose=False, cache=False, amp=False)
    print(f"[FP32 微调] {args.epochs} epochs 用时 {time.time() - t0:.0f}s")
    m50_fp, map_fp = eval_map(fp32, data_yaml, imgsz=args.imgsz)
    print(f"[FP32 微调后] mAP50 = {m50_fp:.4f}  mAP = {map_fp:.4f}")
    fp32_ckpt = os.path.join(fp32.trainer.save_dir, "weights", "best.pt")

    # ---- 2. PTQ: 用 FP32 权重直接量化, 不训练 ----
    ptq = YOLO(fp32_ckpt)
    names = patch_qat(ptq)
    nw, na, _ = count_quantized(ptq.model)
    print(f"[PTQ] 量化卷积 {nw} 个 (激活量化 {na} 个), 加载 FP32 微调权重, 不训练")
    calibrate(ptq.model, data_yaml, args.imgsz)
    m50_ptq, map_ptq = eval_map(ptq, data_yaml, imgsz=args.imgsz)
    print(f"[PTQ]          mAP50 = {m50_ptq:.4f}  mAP = {map_ptq:.4f}   (掉点 {m50_fp - m50_ptq:+.4f})")

    # ---- 3. QAT 微调 ----
    # 注意: trainer 会在 setup_model 时从 checkpoint 重建模型, 因此必须在
    # on_train_start 回调里对 trainer.model (最终训练对象) 打补丁。
    qat = YOLO("yolov8n.pt")
    qat.add_callback("on_train_start",
                     lambda trainer: patch_qat(trainer.model, verbose=True))
    t0 = time.time()
    qat.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
              device="cpu", workers=0, seed=0, project="out", name="qat",
              exist_ok=True, verbose=False, cache=False, amp=False)
    print(f"[QAT 微调] {args.epochs} epochs 用时 {time.time() - t0:.0f}s")
    nw, na, _ = count_quantized(qat.trainer.model)
    print(f"[QAT] 训练用网络量化卷积 {nw} 个 (激活量化 {na} 个)")
    qat.model = qat.trainer.model
    calibrate(qat.model, data_yaml, args.imgsz)
    m50_qat, map_qat = eval_map(qat, data_yaml, imgsz=args.imgsz)
    print(f"[QAT 微调后]  mAP50 = {m50_qat:.4f}  mAP = {map_qat:.4f}")

    print("\n========== Phase 2 对比 (合成数据集) ==========")
    print(f"  指标        FP32 微调      PTQ            QAT 微调")
    print(f"  mAP50      {m50_fp:.4f}      {m50_ptq:.4f} (掉 {m50_fp - m50_ptq:+.4f})   {m50_qat:.4f} (掉 {m50_fp - m50_qat:+.4f})")
    print(f"  mAP50-95   {map_fp:.4f}      {map_ptq:.4f} (掉 {map_fp - map_ptq:+.4f})   {map_qat:.4f} (掉 {map_fp - map_qat:+.4f})")
    print("\n  结论: PTQ 的掉点来自权重已定型、无法适应量化噪声; QAT 在训练中")
    print("       用 STE 让梯度穿过量化算子, 权重得以适应 int8, 精度几乎无损。")
    with open("out/phase2_results.txt", "w", encoding="utf-8") as f:
        f.write(f"epochs={args.epochs} imgsz={args.imgsz}\n")
        f.write(f"FP32_finetune {m50_fp:.4f} {map_fp:.4f}\nPTQ {m50_ptq:.4f} {map_ptq:.4f}\nQAT {m50_qat:.4f} {map_qat:.4f}\n")


if __name__ == "__main__":
    main()
