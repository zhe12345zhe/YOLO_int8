"""真正的 int8 推理部署: TensorRT INT8 engine 构建 + FP32/INT8 精度与延迟对比。

对 QAT 训练权重跑完整 int8 部署链路 (回答: 量化训练到底能否落地为硬件加速):
    1. 权重 -> TensorRT FP32 engine 与 INT8 engine (INT8 构建时内部做 PTQ 校准;
       QAT 权重数值已在 int8 网格上, 校准损失远小于直接 PTQ)
    2. val 集精度对比: mAP50 / mAP50-95 (FP32 engine vs INT8 engine)
    3. 单图延迟对比: warmup + N 次平均 (python 侧含前/后处理, batch=1)
    4. 统计 INT8 engine 中真正 INT8 精度的层数 (证明硬件 int8 kernel 在执行)
可选 --ort: 顺带对比 ONNX Runtime CPU int8 (QDQ 图) 延迟与输出一致性。

用法 (GPU 服务器, 权重/数据就绪后):
    python deploy_int8.py                                            # 自动发现 3 份权重全部跑
    python deploy_int8.py --only wa                                  # 只跑 QAT W+A
    python deploy_int8.py --imgsz 320 --data /root/autodl-tmp/coco-full/data.yaml
    python deploy_int8.py --ort                                      # 附带 onnxruntime 对比
    python deploy_int8.py --n 120 --warmup 20                        # 调 benchmark 次数

结果: out/int8_deploy_results.txt (+ out/int8_bench.png 若 matplotlib 可用)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

from ultralytics import YOLO

CKPT_NAMES = {
    "fp32": ("FP32", "big_fp32", "out/ckpt_fp32_best.pt"),
    "wa": ("QAT_W+A", "big_qat", "out/ckpt_qat_wa_best.pt"),
    "wae": ("QAT_W+A+E", "big_qat_e", "out/ckpt_qat_wae_best.pt"),
}


def find_ckpt(key):
    _, run_name, local = CKPT_NAMES[key]
    cands = [
        os.path.join("runs", "detect", "out", run_name, "weights", "best.pt"),
        local,
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return None


def bench_ms(model, imgsz, n=120, warmup=20):
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(img, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img, verbose=False)
    return (time.perf_counter() - t0) / n * 1000.0


def trt_layer_precision(engine_path):
    """返回 {precision_str: 层数}; 若 API 不可用返回 None。"""
    try:
        import tensorrt as trt
    except ImportError:
        return None
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        prec = {}
        for i in range(engine.num_layers):
            p = str(engine.get_layer(i).precision)
            prec[p] = prec.get(p, 0) + 1
        return prec
    except Exception as e:
        print(f"  [warn] INT8 层统计失败: {e}")
        return None


def export_engine(ckpt, imgsz, data, int8):
    m = YOLO(ckpt)
    path = m.export(format="engine", imgsz=imgsz, device=0,
                    int8=int8, data=data if int8 else None)
    if not path:
        cand = sorted(glob.glob(os.path.join(os.path.dirname(ckpt), "*_int8.engine"
                                             if int8 else "*_fp32.engine")))
        path = cand[-1] if cand else None
    print(f"  {('INT8' if int8 else 'FP32 ')} engine: {path}")
    return path


def run_tensorrt(ckpt, tag, args, out_lines):
    print(f"\n=== TensorRT 对比 [{tag}] ===")
    eng_fp32 = export_engine(ckpt, args.imgsz, args.data, int8=False)
    eng_i8 = export_engine(ckpt, args.imgsz, args.data, int8=True)
    if not eng_fp32 or not eng_i8:
        print("  !! engine 导出失败, 跳过")
        return None

    r32 = YOLO(eng_fp32).val(data=args.data, imgsz=args.imgsz, device=0,
                             verbose=False, plots=False, project="out", name="val_deploy")
    ri8 = YOLO(eng_i8).val(data=args.data, imgsz=args.imgsz, device=0,
                           verbose=False, plots=False, project="out", name="val_deploy")
    m50_32, m95_32 = r32.box.map50, r32.box.map
    m50_8, m95_8 = ri8.box.map50, ri8.box.map

    t32 = bench_ms(YOLO(eng_fp32), args.imgsz, args.n, args.warmup)
    t8 = bench_ms(YOLO(eng_i8), args.imgsz, args.n, args.warmup)

    prec = trt_layer_precision(eng_i8)
    prec_txt = " / ".join(f"{k}:{v}" for k, v in (prec or {}).items()) if prec else "统计不可用"

    line = (f"{tag} mAP50={m50_8:.4f}(F32 {m50_32:.4f}) mAP50-95={m95_8:.4f}(F32 {m95_32:.4f}) "
            f"lat={t8:.1f}ms(F32 {t32:.1f}ms) speedup={t32 / max(t8, 1e-9):.2f}x | INT8层: {prec_txt}")
    print("  " + line)
    out_lines.append(line)
    return dict(tag=tag, m50_32=m50_32, m95_32=m95_32, m50_8=m50_8, m95_8=m95_8,
                t32=t32, t8=t8, prec=prec_txt)


def run_onnxruntime(args, out_lines):
    print("\n=== ONNX Runtime CPU int8 对比 (QDQ 图, 可选) ===")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  未安装 onnxruntime, 跳过 (pip install onnxruntime)")
        return
    qdq = args.qdq or "out/yolov8n_qdq_int8.onnx"
    fp32 = qdq.replace(".onnx", "_fp32.onnx")
    if not (os.path.exists(qdq) and os.path.exists(fp32)):
        print(f"  缺 QDQ/fp32 onnx ({qdq} / {fp32}), 跳过")
        return
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    x = np.random.rand(1, 3, args.imgsz, args.imgsz).astype(np.float32)
    sess_q = ort.InferenceSession(qdq, so, providers=["CPUExecutionProvider"])
    sess_f = ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])

    def bench(sess):
        for _ in range(args.warmup):
            sess.run(None, {"images": x})
        t0 = time.perf_counter()
        for _ in range(args.n):
            sess.run(None, {"images": x})
        return (time.perf_counter() - t0) / args.n * 1000.0

    tq, tf = bench(sess_q), bench(sess_f)
    yq, yf = sess_q.run(None, {"images": x})[0], sess_f.run(None, {"images": x})[0]
    diff = float(np.abs(yq - yf).max())
    line = (f"ORT-CPU: QDQ int8 {tq:.1f}ms vs FP32 {tf:.1f}ms (speedup {tf / max(tq, 1e-9):.2f}x), "
            f"输出最大偏差 {diff:.4f}")
    print("  " + line)
    out_lines.append(line)


def save_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tags = [r["tag"] for r in rows]
    t32 = [r["t32"] for r in rows]
    t8 = [r["t8"] for r in rows]
    m95 = [r["m95_8"] for r in rows]
    x = np.arange(len(tags))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - 0.2, t32, 0.4, label="FP32 engine")
    axes[0].bar(x + 0.2, t8, 0.4, label="INT8 engine", color="orange")
    axes[0].set_xticks(x); axes[0].set_xticklabels(tags)
    axes[0].set_ylabel("latency (ms)"); axes[0].set_title("single-image latency (batch=1)")
    axes[0].legend()
    axes[1].bar(x - 0.2, [r["m95_32"] for r in rows], 0.4, label="FP32 engine")
    axes[1].bar(x + 0.2, m95, 0.4, label="INT8 engine", color="orange")
    axes[1].set_xticks(x); axes[1].set_xticklabels(tags)
    axes[1].set_ylabel("mAP50-95"); axes[1].set_title("val mAP50-95")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("out/int8_bench.png", dpi=130)
    print("\n图已保存: out/int8_bench.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fp32", "wa", "wae"], default=None)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--data", default=None,
                    help="数据集 yaml; 默认: 远端 coco-full, 缺失则 coco128")
    ap.add_argument("--n", type=int, default=120, help="benchmark 次数")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ort", action="store_true", help="附带 onnxruntime CPU 对比")
    ap.add_argument("--qdq", default=None, help="QDQ onnx 路径 (--ort 时)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("!! 无可用 GPU (TensorRT engine 需要 CUDA; --ort 模式可在 CPU 上运行)")
        if not args.ort:
            return
    if args.data is None:
        args.data = "/root/autodl-tmp/coco-full/data.yaml" \
            if os.path.exists("/root/autodl-tmp/coco-full/data.yaml") else "coco128"
    os.makedirs("out", exist_ok=True)

    keys = ["fp32", "wa", "wae"] if not args.only else [args.only]
    out_lines = [f"# int8 deploy @ {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"data={args.data} imgsz={args.imgsz} n={args.n}"]
    rows = []
    for k in keys:
        ckpt = find_ckpt(k)
        if not ckpt:
            print(f"[{k}] 未找到权重, 跳过")
            continue
        print(f"[{k}] 权重: {ckpt}")
        r = run_tensorrt(ckpt, CKPT_NAMES[k][0], args, out_lines)
        if r:
            rows.append(r)
    if args.ort:
        run_onnxruntime(args, out_lines)
    save_plot(rows)

    with open("out/int8_deploy_results.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("\n结果已追加到 out/int8_deploy_results.txt")


if __name__ == "__main__":
    main()