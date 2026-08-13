"""FP32 手搓引擎: 与 int8 引擎同架构 (im2col + GEMM + autograd Function + 每层 python 调度)。

对比 int8 引擎时, 两者的 python 调度结构完全一致, 差异 = 纯 kernel (fp32 matmul vs int8 GEMM)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FP32ConvFn(torch.autograd.Function):
    """前向 = pad + unfold(im2col) + matmul + bias; 反向 = unfold 展开 + matmul (同 int8 引擎结构)。"""

    @staticmethod
    def forward(ctx, x, w, bias, pad, stride):
        N, C, H, W = x.shape
        k = w.shape[2]
        M = w.shape[0]
        xp = F.pad(x, (pad[1], pad[1], pad[0], pad[0]))
        col = F.unfold(xp, kernel_size=k, stride=stride)          # (N, CK, L)
        N, CK, L = col.shape
        x2d = col.permute(0, 2, 1).reshape(N * L, CK)             # (NL, CK)
        w2d = w.reshape(M, CK).transpose(0, 1).contiguous()       # (CK, M)
        y = x2d @ w2d                                             # (NL, M)
        if bias is not None:
            y = y + bias.reshape(1, M)
        Ho = (H + 2 * pad[0] - k) // stride[0] + 1
        Wo = (W + 2 * pad[1] - k) // stride[1] + 1
        ctx.save_for_backward(x2d, w)
        ctx.stride = stride
        ctx.pad = pad
        return y.reshape(N, L, M).permute(0, 2, 1).reshape(N, M, Ho, Wo)

    @staticmethod
    def backward(ctx, grad_y):
        x2d, w = ctx.saved_tensors
        stride, pad = ctx.stride, ctx.pad
        N = grad_y.shape[0]
        M = w.shape[0]
        CK = w.shape[1] * w.shape[2] * w.shape[3]
        dy2d = grad_y.permute(0, 2, 3, 1).reshape(-1, M)          # (NL, M)
        dw = x2d.transpose(0, 1).contiguous() @ dy2d              # (CK, M) 转置+matmul (同 int8)
        dw = dw.reshape(w.shape[1], w.shape[2], w.shape[3], M).permute(3, 0, 1, 2)
        dx2d = dy2d @ w.reshape(M, CK)                            # (NL, CK)
        dx = F.fold(dx2d.reshape(N, -1, CK).transpose(1, 2),
                    output_size=(grad_y.shape[2] * stride[0] + (w.shape[2] - 1) - 2 * pad[0],
                                 grad_y.shape[3] * stride[1] + (w.shape[3] - 1) - 2 * pad[1]),
                    kernel_size=w.shape[2], stride=stride, padding=pad)
        return dx, dw, None, None, None


class FP32Conv2d(nn.Conv2d):
    """替换 nn.Conv2d: 前向/反向 = im2col + fp32 GEMM (与 Int8Conv2d 同架构, 无量化)。"""

    def forward(self, x):
        g = self.groups
        dil = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
        assert g == 1 and dil == 1
        if x.dim() != 4:
            return super().forward(x)
        k = self.kernel_size[0]
        pad = (self.padding[0], self.padding[1])
        stride = (self.stride[0], self.stride[1])
        return _FP32ConvFn.apply(x, self.weight, self.bias, pad, stride)


def patch_fp32_engine(model, skip_head=True, verbose=False):
    net = model.model if hasattr(model, "model") else model
    head_paths = set()
    if skip_head and isinstance(net, nn.Sequential) and len(net) > 0:
        head = net[-1]
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
        m.__class__ = FP32Conv2d
        patched.append(name)
    if verbose:
        print(f"[fp32_engine] 已替换 {len(patched)} 个卷积为 FP32Conv2d (手搓引擎)")
    return patched
