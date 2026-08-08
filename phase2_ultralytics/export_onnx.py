"""导出 QAT 网络为 ONNX, 并用 Netron 可读结构检查量化算子的痕迹。

说明: 本项目 QAT 的量化是"假量化" (fake quant), 权重仍存 FP32,
      int8 只在前向模拟。导出 ONNX 时自定义算子会被展开成基础算子,
      我会统计 ONNX 图里与量化相关的算子 (Round/Clamp/Div/Mul/QuantizeLinear)。

支持两类 checkpoint:
  - 训练态补丁模型 (out/patched_model.pt): 已有 QConv2d 结构, 直接导出
  - ultralytics best.pt (out/qat_e/weights/best.pt 等, strip_optimizer 后):
    重新打补丁 (含 --quant-e 的 E 误差量化 hook) 再导出

注意: E（误差梯度）是反向传播阶段的 backward hook, 不参与前向,
      因此 ONNX 图里只有 W/A 的量化痕迹; 但网络本身就是"W+A+E 训练后的权重"。

用法: python export_onnx.py [--ckpt <path>] [--quant-e] [--imgsz 320] [--out name]
"""
import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO

from qat_patch import patch_qat, count_quantized, QConv2d

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="out/patched_model.pt")
ap.add_argument("--quant-e", action="store_true",
                help="对打补丁的模型注册误差 E 的 backward hook (训练态含 E 量化)")
ap.add_argument("--imgsz", type=int, default=320)
ap.add_argument("--out", default="out/yolov8n_qat_quant.onnx")
args = ap.parse_args()

blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
net = blob["model"] if isinstance(blob, dict) and isinstance(blob.get("model"), nn.Module) else blob
has_patch = any(isinstance(m, QConv2d) for m in net.modules())

y = YOLO("yolov8n.pt")
if has_patch:
    y.model = net                      # 训练态 QConv2d 结构 (含 qw/qa, 无 E hook)
else:
    y = YOLO(args.ckpt)               # ultralytics best.pt: 重新打补丁
    names = patch_qat(y.model, quant_act=True, quant_e=args.quant_e, verbose=True)
    net = y.model

net.eval()
nw, na, ne = count_quantized(net)
print(f"[导出] checkpoint={args.ckpt} QConv2d x {nw} (激活 {na} + 误差E hook {ne})")

dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
with torch.no_grad():
    torch.onnx.export(net, dummy, args.out, opset_version=17, dynamo=False,
                      input_names=["images"], output_names=["output0"])

net.eval()
nw, na, ne = count_quantized(net)
print(f"[导出] checkpoint={args.ckpt} QConv2d x {nw} (激活 {na} + 误差E hook {ne})")

dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
with torch.no_grad():
    torch.onnx.export(net, dummy, args.out, opset_version=17, dynamo=False,
                      input_names=["images"], output_names=["output0"])
print(f"导出完成: {args.out}")

import onnx
m = onnx.load(args.out)
ops = {}
for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("\nONNX 图中算子统计 (量化算子会以这些形式出现):")
for k in sorted(ops, key=lambda x: -ops[x]):
    print(f"  {k:20s} x{ops[k]}")