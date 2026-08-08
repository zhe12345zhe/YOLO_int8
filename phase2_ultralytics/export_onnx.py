"""导出 QAT 网络为 ONNX, 并用 Netron 可读结构检查量化算子的痕迹。

说明: 本项目 QAT 的量化是"假量化" (fake quant), 权重仍存 FP32,
      int8 只在前向模拟。导出 ONNX 时自定义算子会被展开成基础算子,
      我会统计 ONNX 图里与量化相关的算子 (Round/Clamp/Div/Mul/QuantizeLinear),
      并展示其中一处的参数。

用法: python export_onnx.py [--ckpt out/patched_model.pt] [--imgsz 320]
"""
import argparse
import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/patched_model.pt")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--out", default="out/yolov8n_qat_quant.onnx")
args = ap.parse_args()

y = YOLO("yolov8n.pt")
blur = torch.load(args.ckpt, map_location="cpu", weights_only=False)
net = blur["model"]   # QConv2d 结构 (训练态)
net.eval()

dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
with torch.no_grad():
    torch.onnx.export(net, dummy, args.out, opset_version=17, dynamo=False,
                      input_names=["images"], output_names=["output0"])
print(f"导出完成: {args.out}")
print(f"内含 QConv2d x {sum(1 for m in net.modules() if type(m).__name__ == 'QConv2d')}")

import onnx
m = onnx.load(args.out)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("\nONNX 图中算子统计 (量化算子会以这些形式出现):")
for k in sorted(ops, key=lambda x: -ops[x]):
    print(f"  {k:20s} x{ops[k]}")