"""训练过程探针: 证明 QAT 训练循环里量化算子真实执行。

方法: 在 on_train_start 对 trainer.model 打补丁 (45 个 QConv2d),
     每批/每轮打印 quantize_int8 的累计调用次数与量化误差。
        - 每步前向: 45 层 x (1 权重量化 + 1 激活量化) = 90 次调用
        - 若补丁丢失或绕过 => 调 {_} 恒为 0 (即 FP32)

用法: python qat_probe_train.py [--epochs 3] [--imgsz 192]

此外在 on_train_end 会保存补丁后的完整模型快照 (patched_model.pt),
供 qat_prove.py 做结构/消融/网格证据 (绕过 best.pt 的 strip 重建)。
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase1_pytorch"))

import torch
from ultralytics import YOLO
from ultralytics import SETTINGS

import fake_quant as fq
from qat_patch import patch_qat, count_quantized, QConv2d

DATA = "coco128"
DATASETS_DIR = os.path.abspath("datasets")

per_batch_qcalls = []
per_layer_qerr = []


def make_bn():
    def on_start(trainer):
        patch_qat(trainer.model, verbose=True)
        fq.quant_calls_reset()

    def on_batch_end(trainer):
        # 累加本 step 内 quantize 调用次数
        per_batch_qcalls.append(fq.QUANT_CALLS)
        fq.quant_calls_reset()

    def on_train_epoch_end(trainer):
        n = len(per_batch_qcalls)
        if n:
            total = sum(per_batch_qcalls)
            avg = total / n
            per_batch_qcalls.clear()
            # 各层当前量化误差
            qerrs_w, qerrs_a = [], []
            for m in trainer.model.modules():
                if isinstance(m, QConv2d):
                    qerrs_w.append(m.qw.last_qerr)
                    if m.qa is not None:
                        qerrs_a.append(m.qa.last_qerr)
            print(f"    [ep {trainer.epoch}] quantize 调用 {total} 次 / 平均每 batch {avg:.0f} 次"
                  f" | 权重量化误差均值 {sum(qerrs_w)/len(qerrs_w):.4%}"
                  f" | 激活量化误差均值 {sum(qerrs_a)/len(qerrs_a):.4%}")

    return on_start, on_batch_end, on_train_epoch_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=192)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)
    SETTINGS.update(datasets_dir=DATASETS_DIR)

    qat = YOLO("yolov8n.pt")
    on_s, on_b, on_e = make_bn()
    qat.add_callback("on_train_start", on_s)
    qat.add_callback("on_batch_end", on_b)
    qat.add_callback("on_train_epoch_end", on_e)
    qat.train(data=DATA, epochs=args.epochs, imgsz=args.imgsz, batch=8,
              device="cpu", workers=0, seed=0, project="out", name="probe",
              exist_ok=True, verbose=False, cache=False, amp=False)

    net = qat.trainer.model
    nw, na, ne = count_quantized(net)
    print(f"[结束] 训练所用网络仍含 QConv2d x {nw} (此类计数 > 0 即量化补丁存活到训练结束)")
    torch.save({"model": net}, "out/patched_model.pt")   # 保留补丁模型, 供结构证明
    print("已保存补丁模型 out/patched_model.pt")


if __name__ == "__main__":
    main()