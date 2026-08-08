"""数值等价验证: QDQ 量化 ONNX (export_onnx_qdq.py 产物) vs 假量化公式。

从 out/yolov8n_qdq_int8.onnx 中抽出第一个量化子图 (激活 Q->DQ,
权重 DQ, 量化 Conv), 用 onnx 参考实现跑, 与直接用假量化公式
(round(x/s)·s 后卷积) 手算的结果对比, 证明 QDQ 图与 QAT 推理
数值一致 (无量化精度二次损失)。

用法: python verify_qdq_numeric.py [--model out/yolov8n_qdq_int8.onnx]
"""
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="out/yolov8n_qdq_int8.onnx")
args = ap.parse_args()

g = onnx.load(args.model).graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

# 取第一组: 激活 Q/DQ + 权重 DQ + 被量化的 Conv
a_q = next(n for n in g.node if n.op_type == "QuantizeLinear")
a_dq = next(n for n in g.node if n.op_type == "DequantizeLinear"
            and n.input[0] == a_q.output[0])
w_dq = next(n for n in g.node if n.op_type == "DequantizeLinear"
            and n.input[0].endswith(".w_int8"))
conv = next(n for n in g.node if n.op_type == "Conv"
            and n.input[0] == a_dq.output[0] and n.input[1] == w_dq.output[0])

s_a = float(inits[a_q.input[1]])
w8 = inits[w_dq.input[0]]
s_w = inits[w_dq.input[1]]

# 构造独立子图: x -> Q -> DQ 与权重 DQ -> Conv -> y
sub_inits = [
    helper.make_tensor(a_q.input[1], TensorProto.FLOAT, [], [s_a]),
    helper.make_tensor(w_dq.input[0], TensorProto.INT8,
                       list(w8.shape), w8.astype(np.int8).flatten().tolist()),
    helper.make_tensor(w_dq.input[1], TensorProto.FLOAT, list(s_w.shape), s_w.tolist()),
]
if len(conv.input) > 2:      # conv bias (fp32, 不量化)
    bias_name = conv.input[2]
    if bias_name in inits:
        b = inits[bias_name]
        sub_inits.append(helper.make_tensor(bias_name, TensorProto.FLOAT,
                                            list(b.shape), b.tolist()))
    else:
        pnode = next((n for n in g.node if n.op_type == "Constant"
                      and n.output[0] == bias_name), None)
        if pnode is not None:
            b = numpy_helper.to_array(pnode.attribute[0].value)
            sub_inits.append(helper.make_tensor(bias_name, TensorProto.FLOAT,
                                                list(b.shape), b.tolist()))
        else:
            conv.input[:] = conv.input[:2]
a_q.input[0] = "x"   # 输入重定向到子图输入
sub = helper.make_graph(
    nodes=[a_q, a_dq, w_dq, conv],
    name="mini",
    inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
    outputs=[helper.make_tensor_value_info(conv.output[0], TensorProto.FLOAT, [1, 16, 4, 4])],
    initializer=sub_inits,
)
model_sub = helper.make_model(sub, opset_imports=[helper.make_opsetid("", 21)])
ev = ReferenceEvaluator(model_sub)

# 随机输入按 QDQ 图跑
x = np.random.RandomState(0).rand(1, 3, 8, 8).astype(np.float32)
y = ev.run(None, {"x": x})[0]

# 参考实现: 假量化公式
xq = np.clip(np.round(x / s_a), -128, 127) * s_a
wq = w8.astype(np.float32) * s_w.reshape(-1, 1, 1, 1)
attrs = {a.name: list(a.ints) for a in conv.attribute}
bias = None
if len(conv.input) > 2:
    bname = conv.input[2]
    if bname in inits:
        b = inits[bname]
    else:
        b = numpy_helper.to_array(next(n for n in g.node if n.op_type == "Constant"
                                       and n.output[0] == bname).attribute[0].value)
    bias = torch.from_numpy(b)
y_ref = F.conv2d(torch.from_numpy(xq), torch.from_numpy(wq), bias=bias,
                 stride=attrs.get("strides", [1, 1]),
                 padding=attrs.get("pads", [0, 0, 0, 0])[:2]).numpy()

d = np.abs(y - y_ref)
print(f"[验证] 子图输出 max|diff| = {d.max():.2e}  (元素 {y.shape})")
assert d.max() < 1e-3, "QDQ 图与假量化公式不一致!"
print("[结论] QDQ 数值 = 假化量化公式, 一致 (无量化精度二次损失)")