"""量化卷积层: 前向操作数 (W, A) 全部 int8 模拟, 反向经 STE。

反向传播视角 (本文件的核心):
    Y = W_q(x) ⊗ A_q(x)
    权重梯度   ∂L/∂W = A_q ⊗ E      -> 操作数: 量化后的激活 A_q (int8)
    输入梯度   ∂L/∂X = W_q ⊗ E      -> 操作数: 量化后的权重 W_q (int8)
    误差信号   E 经上一层的 q 算子以 STE 直通回传 (fp32 数值, 但建立在量化输出之上)

每个 QuantConv2d 会保存本次前向的 x_q / w_q, 供后续"操作数验证"核对:
autograd 计算出的 w.grad 应与 unfold(x_q) 与 E 的矩阵乘积一致。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fake_quant import ActQuant, WeightQuant


class QuantConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, quant_act=True):
        super().__init__()
        self.conv = conv
        self.wq = WeightQuant(conv.out_channels)
        self.aq = ActQuant() if quant_act else None
        self.saved_x_q = None
        self.saved_w_q = None

    def forward(self, x):
        w_q = self.wq(self.conv.weight)
        x_q = self.aq(x) if self.aq else x
        self.saved_x_q = x_q.detach()
        self.saved_w_q = w_q.detach()
        y = F.conv2d(x_q, w_q, self.conv.bias,
                     self.conv.stride, self.conv.padding,
                     self.conv.dilation, self.conv.groups)
        return y

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias
