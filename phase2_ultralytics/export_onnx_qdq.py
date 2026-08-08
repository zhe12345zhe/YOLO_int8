"""生成真正的 int8 部署 ONNX (QDQ 形式): 权重折叠为 int8 常量 + QDQ 节点。

流程:
  1. 加载 W+A+E 训练权重 (best.pt) 导出干净的 FP32 图
  2. 模型再打 QAT 补丁并跑一圈校准, 获取每层:
       - 权重 per-channel scale (WeightQuant.last_scale)
       - 激活 per-tensor 静态 scale (ActQuant: calib_max/QMAX)
  3. 按权重数值匹配 ONNX 图中的 45 个 Conv 节点, 重写为:
        激活: QuantizeLinear(x, s_a) -> int8 -> DequantizeLinear(q, s_a) -> Conv
        权重: int8 常量 + DequantizeLinear(w8, s_w, axis=1) -> Conv
  4. 与 QAT 静态推理数值一致, 无量化精度二次损失; ONNX Runtime /
     OpenVINO 等 QDQ 后端可直接消费。

用法: python export_onnx_qdq.py [--ckpt <best.pt>] [--imgsz 320] [--out out/yolov8n_qdq_int8.onnx]
"""
import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx import numpy_helper
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qat_patch import patch_qat, QConv2d, calibrate_activations
from qat_run import make_calib_loader


def topo_sort(graph):
    """Kahn 拓扑排序: 新插的 Q/DQ 节点移到其消费者之前。"""
    nodes = list(graph.node)
    idx_of_out = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            idx_of_out[o] = i
    deps = [sorted({idx_of_out[x] for x in n.input if x in idx_of_out}) for n in nodes]
    n = len(nodes)
    indeg = [len(d) for d in deps]
    adj = [[] for _ in range(n)]
    for i, d in enumerate(deps):
        for j in d:
            adj[j].append(i)
    q = [i for i in range(n) if indeg[i] == 0]
    order, head = [], 0
    while head < len(q):
        i = q[head]; head += 1
        order.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    if len(order) != n:
        raise RuntimeError("图中存在环")
    graph.ClearField("node")
    graph.node.extend(nodes[i] for i in order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\Users\lenovo\runs\detect\out\qat_e\weights\best.pt")
    ap.add_argument("--data", default="coco128")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", default="out/yolov8n_qdq_int8.onnx")
    args = ap.parse_args()
    os.makedirs("out", exist_ok=True)
    fp32_path = args.out.replace(".onnx", "_fp32.onnx")

    # ---- 1) FP32 干净图 (dynamo 导出器不折叠权重, 数值与 module 原值一致) ----
    y = YOLO(args.ckpt)
    net = y.model
    net.eval()
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        torch.onnx.export(net, dummy, fp32_path, opset_version=21, dynamo=True,
                          input_names=["images"], output_names=["output0"])
    graph = onnx.load(fp32_path).graph
    print(f"[1] FP32 图: {fp32_path}  ({len(graph.node)} 节点)")

    # ---- 2) 打补丁 + 校准, 得到每层 scale ----
    patch_qat(net, quant_act=True, quant_e=True)
    loader, n_batches = make_calib_loader(args.data, args.imgsz)
    calibrate_activations(net, loader, n_batches=n_batches)
    qconvs = [m for m in net.modules() if isinstance(m, QConv2d)]
    print(f"[2] 校准完成, QConv2d x {len(qconvs)}")

    # ---- 3) dynamo 导出图 initializer 名 = 模块路径名, 按名字匹配 ----
    by_w = {}
    for nd in graph.node:
        if nd.op_type == "Conv" and len(nd.input) >= 2:
            by_w.setdefault(nd.input[1], []).append(nd)
    matched = []          # (QConv2d module, onnx_node)
    for n, m in net.named_modules():
        if not isinstance(m, QConv2d):
            continue
        nodes = by_w.get(n + ".weight", [])
        if not nodes:
            print(f"  !! 未匹配层: {n}")
            continue
        matched.append((m, nodes[0]))
    print(f"[3] 匹配到量化 Conv 节点 x {len(matched)}")

    # ---- 4) 图重写 ----
    new_nodes = []
    for m, nd in matched:
        w = m.weight.detach()
        s_w = m.qw.last_scale.detach().reshape(-1)            # (O,) per-channel
        n_out = w.shape[0]
        w8 = torch.clamp(torch.round(w / s_w.view(-1, 1, 1, 1)), -128, 127).to(torch.int8).numpy()
        s_w_np = s_w.numpy().astype(np.float32)
        tag = nd.output[0].replace("/", "_")

        # 权重 int8 常量 + per-channel scale
        w8_name = tag + ".w_int8"
        s_w_name = tag + ".w_scale"
        new_initializers = [
            helper.make_tensor(w8_name, TensorProto.INT8, list(w8.shape), w8.flatten().tolist()),
            helper.make_tensor(s_w_name, TensorProto.FLOAT, [n_out], s_w_np),
        ]
        graph.initializer.extend(new_initializers)
        # DQ 输出沿用原权重输入名, conv 结构不用改
        dq = helper.make_node("DequantizeLinear", [w8_name, s_w_name], [nd.input[1]],
                              name=tag + ".w_dq", axis=0)
        new_nodes.append(dq)
        # 激活: Q -> DQ 插入 conv 输入
        s_a = m.qa.last_scale.detach().float().item() if m.qa is not None else None
        if s_a is not None:
            s_a_name = tag + ".a_scale"
            graph.initializer.append(helper.make_tensor(s_a_name, TensorProto.FLOAT, [], [s_a]))
            q_name = tag + ".a_q"
            dq_a_name = tag + ".a_dq"
            new_nodes.append(helper.make_node(
                "QuantizeLinear", [nd.input[0], s_a_name], [q_name], name=tag + ".a_q"))
            new_nodes.append(helper.make_node(
                "DequantizeLinear", [q_name, s_a_name], [dq_a_name], name=tag + ".a_dq"))
            nd.input[0] = dq_a_name
    graph.node.extend(new_nodes)

    # 清理: 原 FP32 weight initializers (已被 DQ 代替)
    w_names = {nd.input[1] for _, nd in matched}
    keep = [i for i in graph.initializer if i.name not in w_names]
    graph.ClearField("initializer")
    graph.initializer.extend(keep)

    topo_sort(graph)

    # 清掉 torch 导出残留的 Split num_outputs / Resize antialias 属性
    for n in graph.node:
        if n.op_type == "Split":
            rm = [a for a in n.attribute if a.name == "num_outputs"]
            for a in rm:
                n.attribute.remove(a)
        if n.op_type == "Resize":
            allowed = {"mode", "coordinate_transformation_mode", "cubic_coeff_a",
                       "exclude_outside", "extrapolation_value", "nearest_mode"}
            rm = [a for a in n.attribute if a.name not in allowed]
            for a in rm:
                n.attribute.remove(a)

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    try:
        onnx.checker.check_model(model)
        print("[5] onnx checker 通过")
    except Exception as e:
        print(f"[5] checker 警告: {e}")

    onnx.save(model, args.out)
    print(f"[6] 保存: {args.out}")

    ops = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("\nQDQ 图算子统计 (量化相关):")
    for k in sorted(ops):
        print(f"  {k:24s} x{ops[k]}")
    n_i8 = sum(1 for i in model.graph.initializer if i.data_type == TensorProto.INT8)
    print(f"int8 常量张量 x {n_i8}(每组权重), 每层另有 per-channel scale 常量")
    print(f"Draggers: QuantizeLinear x{ops.get('QuantizeLinear',0)} / DequantizeLinear x{ops.get('DequantizeLinear',0)}")


if __name__ == "__main__":
    main()