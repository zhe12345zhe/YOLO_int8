"""Phase 2 大数据集实验: COCO2017 子集 (train 1200 / val 500) 上 FP32 vs QAT(W+A) vs QAT(W+A+E)。

用法:
    python qat_run_big.py --stage fp32      # 单跑 FP32
    python qat_run_big.py --stage wa        # 单跑 QAT W+A
    python qat_run_big.py --stage wae       # 单跑 QAT W+A+E
    python qat_run_big.py --stage all       # 依次全部 (默认 15 epochs)
支持断点续训: 对应 out/<name>/weights/last.pt 存在时自动 resume。
"""
import argparse
import os
import time

import torch
from ultralytics import YOLO, SETTINGS

from qat_patch import patch_qat, count_quantized, calibrate_activations
from qat_run import calibrate, eval_map

DATA = "datasets/coco-big/data.yaml"
STAGES = {"fp32": ("big_fp32", None, "FP32"),
          "wa": ("big_qat", False, "QAT_W+A"),
          "wae": ("big_qat_e", True, "QAT_W+A+E")}


def find_ckpt(name, suffix="best.pt"):
    """ultralytics 8.x 实际把权重写到 runs/detect/out/<name>/weights/, 兼容两处路径。"""
    cands = [
        os.path.join("runs", "detect", "out", name, "weights", suffix),
        os.path.join("out", name, "weights", suffix),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def is_resumable(ckpt_path):
    """last.pt 需带 optimizer/epoch 状态才能 resume (训练完成时 ultralytics 会 strip 掉)。"""
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return "optimizer" in ck and ck.get("epoch", -1) >= 0
    except Exception:
        return False


def make_early_stop(name, patience=5, tol=2e-3):
    """box loss 停滞早停: 连续 N 个 epoch box loss 变化 < tol 则 trainer.stop。"""
    state = {"count": 0}

    def cb(trainer):
        import csv as _csv
        import os as _os
        f = _os.path.join("runs", "detect", "out", name, "results.csv")
        if not _os.path.exists(f):
            return
        rows = list(_csv.DictReader(open(f)))
        if len(rows) < patience + 1:
            return
        recent = [float(r["train/box_loss"]) for r in rows[-patience - 1:-1]]
        delta = max(recent) - min(recent)
        if delta < tol:
            state["count"] += 1
            print(f"[early-stop] box loss 停滞 {state['count']}/2 (delta {delta:.5f})")
            if state["count"] >= 2:
                trainer.stop = True
                print("[early-stop] 触发停止")
        else:
            state["count"] = 0

    return cb


def run_one(name, quant_e, tag, args):
    last_ckpt = find_ckpt(name, "last.pt")
    best_ckpt = find_ckpt(name, "best.pt")
    if last_ckpt and is_resumable(last_ckpt):
        model = YOLO(last_ckpt, task="detect")
        can_resume = True
        print(f"[{tag}] 断点续训: {last_ckpt}")
    elif best_ckpt:
        model = YOLO(best_ckpt, task="detect")
        can_resume = False
        print(f"[{tag}] 训练已完成, 直接评估: {best_ckpt}")
    else:
        model = YOLO("yolov8n.pt")
        can_resume = False
    if quant_e is not None:
        model.add_callback("on_train_start",
                           lambda trainer: patch_qat(trainer.model, quant_act=True,
                                                     quant_e=quant_e, verbose=True))
    if args.early_stop:
        model.add_callback("on_train_epoch_end", make_early_stop(name))
    trained = False
    if can_resume or not best_ckpt:
        t0 = time.time()
        kw = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device="0", workers=8, seed=0, project="out", name=name,
                    exist_ok=True, verbose=False, cache=False, amp=False,
                    resume=can_resume)
        if args.lr0 is not None:
            kw["lr0"] = args.lr0
            kw["warmup_epochs"] = args.warmup_epochs
        if args.optimizer is not None:
            kw["optimizer"] = args.optimizer
        model.train(**kw)
        print(f"[{tag} 微调] 用时 {time.time() - t0:.0f}s")
        trained = True
    net = model.trainer.model if trained else model.model
    if quant_e is not None and can_resume is False and trained is False:
        # 直接评估已有权重时, 也需挂上量化补丁再校准
        patch_qat(net, quant_act=True, quant_e=quant_e, verbose=True)
    if quant_e is not None and trained:
        nw, na, ne = count_quantized(net)
        print(f"[{tag}] 量化卷积 {nw} 个 (激活 {na} + 误差E {ne})")
    model.model = net
    calibrate(model.model, args.data, args.imgsz)
    m50, m95 = eval_map(model, args.data, imgsz=args.imgsz)
    print(f"[{tag} 微调后] mAP50 = {m50:.4f}  mAP50-95 = {m95:.4f}")
    return m50, m95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DATA)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--lr0", type=float, default=None, help="initial lr (large batch needs linear scaling)")
    ap.add_argument("--warmup-epochs", type=float, default=5.0, help="warmup epochs")
    ap.add_argument("--early-stop", action="store_true", help="box loss 停滞早停")
    ap.add_argument("--optimizer", type=str, default=None, help="optimizer (auto 会用 LRFinder 覆盖 lr0, 显式指定如 SGD 才让 lr0 生效)")
    ap.add_argument("--stage", type=str, default="all",
                    choices=["all", "fp32", "wa", "wae"])
    args = ap.parse_args()

    torch.set_num_threads(max(4, torch.get_num_threads()))
    os.makedirs("out", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)
    SETTINGS.update(datasets_dir=os.path.abspath("datasets"))

    results = {}
    order = ["fp32", "wa", "wae"] if args.stage == "all" else [args.stage]
    for s in order:
        name, qe, tag = STAGES[s]
        results[tag] = run_one(name, qe, tag, args)

    print("\n========== COCO-big (train 1200 / val 500) 对比 ==========")
    print(f"  方案                mAP50       mAP50-95      vs FP32(mAP50-95)")
    if "FP32" in results:
        f50, f95 = results["FP32"]
        print(f"  FP32 微调(上界)    {f50:.4f}      {f95:.4f}")
        for tag in ("QAT_W+A", "QAT_W+A+E"):
            if tag in results:
                m50, m95 = results[tag]
                print(f"  {tag:16s} {m50:.4f}      {m95:.4f}      {m95 - f95:+.4f}")
    with open("out/phase2_big_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- run @ {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"data={args.data} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}\n")
        for tag in ("FP32", "QAT_W+A", "QAT_W+A+E"):
            if tag in results:
                m50, m95 = results[tag]
                f.write(f"{tag} {m50:.4f} {m95:.4f}\n")


if __name__ == "__main__":
    main()