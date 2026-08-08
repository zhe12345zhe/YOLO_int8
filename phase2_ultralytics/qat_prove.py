"""证明 QAT 训练真的在量化 (而非退化为 FP32 微调)。

四层证据:
  [1] 结构证据: 加载训练产出的 QAT checkpoint, 检查其中的卷积类型
              必须全部是 QConv2d (每个带 qw/qa 量化器)。
  [2] 运行证据: 前向埋点统计 quantize_int8 的实际调用次数
              (校准/推理时每个量化张量一次)。
  [3] 消融证据: 同一份权重, "量化开 / 关" 两种推理模式评估,
              与 FP32 权重对比, 形成 2x2 矩阵。
  [4] 网格证据: 权重的 int8 量化误差 max|W - W_q|/max|W|:
              QAT 训练过的权重误差应显著小于普通 FP32 权重
              (权重被 STE 逐步拉向量化网格)。

用法: python qat_prove.py [--qat <best.pt>] [--fp32 <best.pt>] [--data coco128] [--imgsz 320]
"""
import argparse
import os
import statistics
from collections import Counter

import torch
from ultralytics import YOLO
from ultralytics.data import YOLODataset, build_dataloader
from ultralytics.data.utils import check_det_dataset

from qat_patch import patch_qat, count_quantized, calibrate_activations, QConv2d
import fake_quant as fq


def make_calib_loader(data_yaml, imgsz, batch=16):
    data = check_det_dataset(data_yaml)
    ds = YOLODataset(img_path=data["val"], imgsz=imgsz, batch_size=batch, augment=False,
                     rect=False, stride=32, data=data, single_cls=False)
    loader = build_dataloader(ds, batch=batch, workers=0, shuffle=False)
    return loader, (len(ds) + batch - 1) // batch


def eval_map(model, data, imgsz):
    r = model.val(data=data, imgsz=imgsz, device="cpu", verbose=False,
                  plots=False, project="out", name="prove_val", exist_ok=True)
    return r.box.map50, r.box.map


def weight_qerrs(model):
    """各量化卷积的权重 int8 量化误差 max|W-Wq|/max|W|。"""
    errs = []
    for m in model.modules():
        if isinstance(m, QConv2d):
            m.qw(m.weight.detach())
            errs.append(m.qw.last_qerr)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qat", default="out/patched_model.pt")
    ap.add_argument("--qat50", default=r"C:\Users\lenovo\runs\detect\out\qat\weights\best.pt")
    ap.add_argument("--fp32", default=r"C:\Users\lenovo\runs\detect\out\fp32\weights\best.pt")
    ap.add_argument("--data", default="coco128")
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)

    print("=" * 70)
    print("[1] 结构证据: 训练产出的 checkpoint 里卷积是什么类?")
    blur = torch.load(args.qat, map_location="cpu", weights_only=False)
    qat = YOLO("yolov8n.pt")
    qat.model = blur["model"]   # 训练中带补丁的网络 (QConv2d)
    qtypes = Counter(type(m).__name__ for m in qat.model.modules() if isinstance(m, torch.nn.Conv2d))
    print(f"    QAT checkpoint 的 Conv2d 类分布: {dict(qtypes)}")
    nw, na, ne = count_quantized(qat.model)
    print(f"    -> QConv2d x {nw} 个 (每个内含权重量化 qw + 激活量化 qa)")

    fp32 = YOLO(args.fp32)
    ftypes = Counter(type(m).__name__ for m in fp32.model.modules() if isinstance(m, torch.nn.Conv2d))
    print(f"    FP32 checkpoint 的 Conv2d 类分布: {dict(ftypes)}")
    patch_qat(fp32.model)   # PTQ 对照: 给 FP32 权重临时挂量化

    print("=" * 70)
    print("[3] 消融证据: 同一权重, 量化开/关 推理对比 (2x2 矩阵)")
    loader, n = make_calib_loader(args.data, args.imgsz)
    qat50 = YOLO(args.qat50)                # 50 轮训练的 QAT 权重 (best.pt 被 strip 成 plain)
    patch_qat(qat50.model)
    rows = []
    for model, tag in [(qat, "QAT-3ep"), (qat50, "QAT-50ep"), (fp32, "FP32-50ep")]:
        fq.quant_calls_reset()
        calibrate_activations(model.model, loader, n_batches=n)
        c = fq.QUANT_CALLS
        fq.QUANT_ENABLED = True
        m50_on, m_on = eval_map(model, args.data, args.imgsz)
        fq.QUANT_ENABLED = False
        m50_off, m_off = eval_map(model, args.data, args.imgsz)
        fq.QUANT_ENABLED = True
        rows.append((tag, m50_on, m_on, c, m50_off, m_off))
        print(f"RESULT {tag}: ON={m50_on:.4f}/{m_on:.4f} ({c} calls) | OFF={m50_off:.4f}/{m_off:.4f}")
    with open("out/prove_ablation.csv", "w", encoding="utf-8") as f:
        f.write("tag,mAP50_on,mAP50_95_on,quant_calls,mAP50_off,mAP50_95_off\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("    -> 已写入 out/prove_ablation.csv")

    print("=" * 70)
    print("[4] 网格证据: 权重 int8 量化误差 max|W-Wq|/max|W|")
    eq = weight_qerrs(qat.model)
    eq50 = weight_qerrs(qat50.model)
    ef = weight_qerrs(fp32.model)
    print(f"    QAT-3ep  权重: 均值 {statistics.mean(eq):.4%}  最大 {max(eq):.4%}")
    print(f"    QAT-50ep 权重: 均值 {statistics.mean(eq50):.4%}  最大 {max(eq50):.4%}")
    print(f"    FP32-50ep 权重: 均值 {statistics.mean(ef):.4%}  最大 {max(ef):.4%}")
    print(f"    -> 若 QAT 均值 < FP32 均值: 训练中权重被拉向 int8 网格")

    print("=" * 70)
    print("判定标准:")
    print("  [1] 若卷积全是 plain Conv2d            -> QAT 就是 FP32 (补丁被训练器覆盖)")
    print("  [2] 若量化开关结果完全相同             -> 量化根本没参与推理")
    print("  [1]+[2]+[4] 全部通过 => QAT 是真量化训练")


if __name__ == "__main__":
    main()