"""给 ultralytics YOLO 模型插入 int8 假量化 (QAT) 补丁。

做法: 通过"类替换" (m.__class__ = QConv2d) 给每个 Conv2d 挂上量化器,
不改变任何 state_dict 键名, 因此可以无缝加载 yolo8n.pt 预训练权重。

补丁位置:
    1. 权重量化 (per-channel int8, 反向 STE)
    2. 激活量化 (per-tensor int8, 反向 STE)
检测头 (Detect) 保持 fp32: 损失信号通道不量化 (与业界主流做法一致)。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase1_pytorch"))
from fake_quant import ActQuant, WeightQuant


class QConv2d(nn.Conv2d):
    """替换 nn.Conv2d 的类: 前向 = conv(aq(x), wq(W)), 反向经 STE 回传。"""

    def forward(self, x):
        w_q = self.qw(self.weight)
        x_q = self.qa(x) if self.qa is not None else x
        return F.conv2d(x_q, w_q, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def patch_qat(model, quant_act=True, verbose=False):
    """model: ultralytics YOLO 实例 (或 DetectionModel). 返回被 patch 的 conv 名字列表。"""
    net = model.model if hasattr(model, "model") else model
    head = net[-1] if isinstance(net, nn.Sequential) else None
    head_paths = set()
    if head is not None:
        for n, m in net.named_modules():
            if m is head:
                head_paths.add(n)
                break
    patched = []
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if head_paths and any(name.startswith(hp) for hp in head_paths):
            continue
        m.__class__ = QConv2d
        m.qw = WeightQuant(m.out_channels)
        m.qa = ActQuant() if quant_act else None
        patched.append(name)
    if verbose:
        nw, na = count_quantized(net)
        print(f"[QAT] 已对 {nw} 个卷积打补丁 (激活量化 {na} 个)")
    return patched


def count_quantized(net):
    n_w = n_a = 0
    for m in net.modules():
        if isinstance(m, QConv2d):
            n_w += 1
            n_a += 1 if m.qa is not None else 0
    return n_w, n_a


def calibrate_activations(net, loader, n_batches=None):
    """静态校准: 在 loader 上跑一遍前向, 收集各层激活最大值并冻结为固定 scale。

    等价真实推理引擎的 PTQ 静态校准 (用校准集统计激活范围)。
    loader 为 ultralytics 的 InfiniteDataLoader, 迭代 n_batches 次即一轮。
    """
    net.eval()
    qas = [m.qa for m in net.modules() if isinstance(m, QConv2d) and m.qa is not None]
    for qa in qas:
        qa.static = True
        qa.start_calibrating()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["img"] if isinstance(batch, dict) else batch[0]
            imgs = imgs.float()
            if imgs.max() > 1.5:      # 校准集为 uint8(0-255) 时归一化, 与训练一致
                imgs = imgs / 255.0
            net(imgs)
            if n_batches is not None and i + 1 >= n_batches:
                break
    for qa in qas:
        qa.stop_calibrating()
    print(f"[校准] 冻结 {len(qas)} 个激活量化器的静态 scale")
