"""int8 假量化算子 (fake quantization) 与 STE 反向传播。

核心概念
--------
1. 前向: 真实地模拟 int8 量化        x_q = (clamp(round(x / s + zp), -128, 127) - zp) * s
2. 反向: 两种选择
   - STE (Straight-Through Estimator): 把 round 的导数伪造为恒等,
     梯度以 fp32 直通; 超出量化范围的位置梯度截断为 0。
   - TrueGrad: round 的真实导数几乎处处为 0, 会导致训练停滞,
     仅用于对照实验证明 STE 的必要性。

注意: 这里"操作数"的量化 = 参与乘法/卷积的数值 (W, A) 是量化后的 int8 值;
      STE 只是 q 算子自身的伪梯度, 与操作数是否量化无关。
"""
import torch
import torch.nn as nn

QMIN, QMAX = -128, 127

# 全局开关与调用计数: 用于"证明量化真在运行"的埋点
#   QUANT_ENABLED 设为 False 时, 所有假量化算子退化为恒等 (fp32 直通), 用于对照评估
#   QUANT_CALLS    统计 quantize_int8 被调用的次数 (每次前向每个量化张量 +1)
QUANT_ENABLED = True
QUANT_CALLS = 0


def quantize_int8(x, scale, zero_point=0.0, ste=True):
    global QUANT_CALLS
    if not QUANT_ENABLED:
        return x
    QUANT_CALLS += 1
    if ste:
        return _FakeQuantizeSTE.apply(x, scale, zero_point)
    return _FakeQuantizeTrue.apply(x, scale, zero_point)


def quant_calls_reset():
    global QUANT_CALLS
    QUANT_CALLS = 0


class _FakeQuantizeSTE(torch.autograd.Function):
    """STE 版量化算子: 前向走真实量化, 反向梯度直通 (截断范围外)。"""

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        ctx.save_for_backward(x, scale, torch.as_tensor(zero_point, dtype=torch.float32, device=x.device))
        xq = torch.clamp(torch.round(x / scale + zero_point), QMIN, QMAX)
        return (xq - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point = ctx.saved_tensors
        lo = (QMIN - zero_point) * scale
        hi = (QMAX - zero_point) * scale
        mask = (x >= lo) & (x <= hi)
        return grad_output * mask, None, None


class _FakeQuantizeTrue(torch.autograd.Function):
    """真实导数版: round 的导数几乎处处为 0, 用于对照实验。"""

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        xq = torch.clamp(torch.round(x / scale + zero_point), QMIN, QMAX)
        return (xq - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), None, None


class WeightQuant(nn.Module):
    """权重 int8 量化: per-output-channel 对称量化, scale = max|W| / 127。

    前向把权重真实量化 (int8 值, fp32 存储); 反向经 STE 流动,
    因此梯度更新的是"量化后的权重"等价路径, 训练出的权重天然适配 int8。
    """

    def __init__(self, out_channels, per_channel=True, ste=True):
        super().__init__()
        self.out_channels = out_channels
        self.per_channel = per_channel
        self.ste = ste
        self.last_scale = None
        self.last_qerr = 0.0

    def forward(self, w):
        if self.per_channel:
            scale = w.abs().amax(dim=(1, 2, 3)) / QMAX  # (C_out,)
            scale = scale.clamp(min=1e-8).reshape(-1, 1, 1, 1)
        else:
            scale = w.abs().max().clamp(min=1e-8)
        self.last_scale = scale.detach()
        w_q = quantize_int8(w, scale, 0.0, ste=self.ste)
        self.last_qerr = (w - w_q).abs().max().item() / w.abs().max().item()
        return w_q

    def dequant_weight(self, w):
        return quantize_int8(w, self.last_scale, 0.0, ste=False)


class ActQuant(nn.Module):
    """激活 int8 量化: per-tensor 对称量化。

    两种校准模式:
      dynamic (训练/QAT): scale 取当前 batch 最大绝对值, 逐 batch 变化
      static  (部署/PTQ): 先在校准集上收集各层激活最大值 (calibrate),
                         之后冻结为固定 scale (等价真实推理引擎的静态校准)
    """

    def __init__(self, static=False):
        super().__init__()
        self.static = static
        self.ste = True
        self.calibrating = False
        self.register_buffer("calib_max", torch.tensor(-1.0))
        self.last_scale = None
        self.last_qerr = 0.0

    def forward(self, x):
        if self.static and self.calib_max > 0:
            s = self.calib_max / QMAX
        else:
            s = x.detach().abs().amax() / QMAX
        s = s.clamp(min=1e-8)
        if self.calibrating:
            xmax = x.detach().abs().amax()
            if xmax > self.calib_max:
                self.calib_max.copy_(xmax)
        self.last_scale = s.detach()
        x_q = quantize_int8(x, s, 0.0, ste=self.ste)
        self.last_qerr = (x - x_q).abs().max().item() / (x.abs().max().item() + 1e-8)
        return x_q

    def start_calibrating(self):
        self.calibrating = True
        self.calib_max.fill_(-1.0)

    def stop_calibrating(self):
        self.calibrating = False
