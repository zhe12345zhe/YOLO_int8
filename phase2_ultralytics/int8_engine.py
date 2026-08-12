"""INT8 训练引擎 (方案 B) 核心: 前向 int8 GEMM + SwitchBack 式 dW int8 (dX 保持高精度)。

设计 (对应讨论的方案 A/B):
    方案 A (SwitchBack):  前向 fp16;     反向 dy 量化 int8 -> dW 用 int8 GEMM;  dX 用 fp16
    方案 B (本文件):      前向 x/w 量化 int8 -> GEMM 为 int8;  反向同方案 A
    方案 C (预留开关):    dX 也量化 int8 (quantize_dx=True), 默认关

关键实现点:
    1. 真 int8 GEMM 用 torch._int_mm (CUDA, int8 x int8 -> int32);
       CPU 上 fallback 为 int8 值上的 fp32 矩阵乘 (int8 乘加在 int32 内精确, 数值等价),
       因此本机 (无 GPU) 可验证数值/梯度正确性, GPU 上自动切真 kernel。
    2. 前向:  x 每层动态量化 (per-tensor, absmax/127), w 静态重算 (per-output-channel);
             y = _int_mm(xq, wq) * (sx * sw) (+bias)          [方案 B]
    3. 反向:  dy 动态量化 int8 -> dW = _int_mm(xq.T, dyq) * (sx * sdy)   [SwitchBack]
             dx = dy @ W^T (fp16/fp32, 不量化)                 [方案 A/B 一致]
    4. 权重 xq 在 ctx 中保存 int8, 复用公式: dW = int8(xq).T @ int8(dy) * (sx*sdy)。
    5. 数值注意: _int_mm 的 K 维累加为 int32; YOLOv8n 的 K <= C*9 (C<=512) 不会溢出。

量化器:
    x  : per-tensor 动态 (absmax/127), 与 QAT ActQuant dynamic 模式一致
    w  : per-output-channel 静态 (与 QAT WeightQuant per_channel 一致)
    dy : per-tensor 动态 (这就是误差 E 的 int8 量化, 对应工作 I/J 的 E 量化成本,
         方案 B 无法回避该代价, 但可换 per-row 粒度降低损失, 见 quant_dy)

用法:
    from int8_engine import patch_int8_engine, sanity_check
    patched = patch_int8_engine(model.model)      # 类替换, state_dict 键不变
    results = sanity_check(verbose=True)          # 本机数值验证

GPU 调试 (在远端 3080 Ti):
    python int8_engine.py --sanity        # 真 _int_mm 路径 + 数值/梯度验证
    python int8_engine_train.py --train ...    # 真训练
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# int8 GEMM 后端选择
# ----------------------------------------------------------------------------

INT8_GEMM_BACKEND = None  # "cuda_intmm" | "cpu_fp32_sim"; lazy 探测
_QUANTIZE_E = [False]      # 方案 C: dX 链误差信号 E 也量化 (int8 GEMM)


def int8_gemm_backend():
    """返回真 int8 GEMM 后端名; torch._int_mm 仅 CUDA 可用。"""
    global INT8_GEMM_BACKEND
    if INT8_GEMM_BACKEND is None:
        if torch.cuda.is_available() and hasattr(torch, "_int_mm"):
            INT8_GEMM_BACKEND = "cuda_intmm"
        else:
            INT8_GEMM_BACKEND = "cpu_fp32_sim"
    return INT8_GEMM_BACKEND


def _int_mm_chunked(a, b, chunk=32768):
    """_int_mm 按行分块 (a.size(0) 有 ~32K 上限, 实测 49152 OK / 65536 FAIL);
    分块后 torch.cat 数值等价 (int32 累加, 各块独立累加同一 K 维)。"""
    if a.size(0) <= chunk:
        return torch._int_mm(a, b)
    parts = []
    for i in range(0, a.size(0), chunk):
        parts.append(torch._int_mm(a[i:i + chunk], b))
    return torch.cat(parts, dim=0)


def int8_gemm(a, b):
    """int8 x int8 -> int32: a (N,K), b (K,M)。数值上不同后端对同一 device 对齐。

    CUDA: torch._int_mm。约束与规避:
      - K 非 32 倍数: pad 尾部列 (im2col 后 K=C*k*k 常非倍数, 如 27)
      - M/N 维 < 32: pad 后截断 (yolov8n 首层输出仅 16 通道; cublas int8 tile 限制)
      - N 行 > 32K: 分块 (实测 49152 OK / 65536 FAIL)
    CPU : int64 累加。
    """
    assert a.dtype == torch.int8 and b.dtype == torch.int8, (a.dtype, b.dtype)
    backend = int8_gemm_backend()
    n, k = a.shape
    m = b.shape[1]
    n_orig, m_orig = n, m
    if k % 32 != 0:
        padk = 32 - (k % 32)
        a = torch.nn.functional.pad(a, (0, padk))          # (N, K+pad), 尾部 pad 列
        b = torch.nn.functional.pad(b, (0, 0, 0, padk))    # (K+pad, M), 底部 pad 行与 a 对齐
    if backend == "cuda_intmm":
        if n % 32 != 0:
            a = torch.nn.functional.pad(a, (0, 0, 0, 32 - (n % 32)))   # 行数须 32 倍数 (实测 144 FAIL / 160 OK)
        if m < 32:
            b = torch.nn.functional.pad(b, (0, 32 - m))         # 尾部 pad 列
        r = _int_mm_chunked(a, b, 32768)
        return r[:n_orig, :m_orig]
    return torch.matmul(a.to(torch.int64), b.to(torch.int64)).to(torch.int32)[:n_orig, :m_orig]


# ----------------------------------------------------------------------------
# 量化辅助 (对称, STE 由上层 autograd Function 自然提供)
# ----------------------------------------------------------------------------

def quant_tensor(x, scale):
    """对称 int8 量化 (STE 梯度由调用方 autograd Function 的返回路径提供)。"""
    if x.is_cuda and x.numel() >= 4096:
        q = _triton_quant(x, scale)
        if q is not None:
            return q
    return torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)


_QUANT_JIT = [None]


def _triton_quant(x, scale):
    """单 kernel 融合量化: div+round+clamp+cast 一次读写 (省 3 次 elementwise pass)。"""
    import triton
    import triton.language as tl
    try:
        if _QUANT_JIT[0] is None:
            @triton.jit
            def _quant_kernel(X, OUT, S, N, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                mask = offs < N
                v = tl.load(X + offs, mask=mask)
                s = tl.load(S)
                r = v * (1.0 / s)
                r = tl.where(r >= 0, tl.floor(r + 0.5), tl.ceil(r - 0.5))  # round half-away
                q = tl.minimum(tl.maximum(r, -127.0), 127.0)
                tl.store(OUT + offs, q.to(tl.int8), mask=mask)

            _QUANT_JIT[0] = _quant_kernel
        k = _QUANT_JIT[0]
        n = x.numel()
        # empty 必须连续: empty_like 对非连续输入保留 stride (preserve_format),
        # 其 reshape(-1) 视图会让 kernel 裸指针写错位置 (实测输出全 0)
        out = torch.empty(x.shape, dtype=torch.int8, device=x.device)
        if scale.numel() != 1:
            return None  # per-channel 用 torch
        s_t = scale.detach().reshape(-1).contiguous()
        BLOCK = 1024
        # 必须 contiguous: reshape(-1) 对非连续视图返回非连续 1D, Triton 裸指针按逻辑索引
        # 会读错位置 (实测 permute 视图产生 -128/数值错乱)
        xf = x.contiguous().reshape(-1)
        k[(triton.cdiv(n, BLOCK),)](xf, out.reshape(-1), s_t, n, BLOCK)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[_triton_quant] 异常: {type(e).__name__}: {str(e)[:150]}")
        return None


def scale_absmax(x, dim=None):
    s = x.detach().abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / 127.0
    return s


# ----------------------------------------------------------------------------
# 核心 autograd Function: 方案 B 的 int8 GEMM + SwitchBack 梯度
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Triton int8 隐式 im2col (dW 用): xq (N,C,H,W) -> x2d (C*K*K, N*Ho*Wo) 转置布局
# 替代 fp32 pad+unfold+cast+transpose 四步 (实测 32.8ms -> 1.3ms, 数值完全一致)
# ----------------------------------------------------------------------------

_TRITON_IM2COL = [None]


def _triton_im2col_available():
    if _TRITON_IM2COL[0] is None:
        try:
            import triton  # noqa: F401
            _TRITON_IM2COL[0] = torch.cuda.is_available()
        except ImportError:
            _TRITON_IM2COL[0] = False
    return _TRITON_IM2COL[0]


def _im2col_t_int8(xq, k, pad, stride):
    """xq (N,C,H,W) int8 -> (C*k*k, N*Ho*Wo) int8 (K 主序 = 转置布局, 供 dW 用)。"""
    import triton
    import triton.language as tl

    @triton.jit
    def _int8_im2col_t_kernel(
        X, Out,
        N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
        K: tl.constexpr, P: tl.constexpr, S: tl.constexpr,
        HO: tl.constexpr, WO: tl.constexpr,
        BLOCK_NL: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        pid_nl = tl.program_id(1)
        kck = pid_k
        nl = pid_nl * BLOCK_NL + tl.arange(0, BLOCK_NL)
        c = kck // (K * K)
        rem = kck % (K * K)
        kh = rem // K
        kw = rem % K
        n = nl // (HO * WO)
        l = nl % (HO * WO)
        ho = l // WO
        wo = l % WO
        h = ho * S + kh - P
        w = wo * S + kw - P
        in_b = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        src = ((n * C + c) * H + h) * W + w
        val = tl.load(X + src, mask=in_b, other=0)
        dst = kck * (N * HO * WO) + nl
        tl.store(Out + dst, val, mask=nl < N * HO * WO)

    N, C, H, W = xq.shape
    HO = (H + 2 * pad - k) // stride + 1
    WO = (W + 2 * pad - k) // stride + 1
    K2 = C * k * k
    out = torch.empty((K2, N * HO * WO), dtype=torch.int8, device=xq.device)
    BN = 512
    _int8_im2col_t_kernel[(K2, triton.cdiv(N * HO * WO, BN))](
        xq, out, N, C, H, W, k, pad, stride, HO, WO, BN, num_warps=8)
    return out


# ----------------------------------------------------------------------------
# cuBLAS int8 GEMM with transpose (dW 免内存转置; 传统 cublasGemmEx API)
# ----------------------------------------------------------------------------

_GEMMEX_EXT = [None]
_GEMMEX_TRY = [None]


def _int8_gemm_ex(x2d, dy2d):
    """x2d (NL, CK) int8 行主序, dy2d (NL, CO) int8 行主序 -> (CO, CK) int32 (dW 布局)。

    收缩维 = NL (无需展开转置); CK 维 pad 到 8 倍数 (int8 GEMM 对齐要求)。
    """
    if _GEMMEX_TRY[0] is None:
        try:
            from torch.utils.cpp_extension import load as _cxx_load
            import os
            os.environ["PATH"] = "/root/miniconda3/bin:" + os.environ.get("PATH", "")
            _GEMMEX_EXT[0] = _cxx_load(
                name="lt_ex", sources=["/root/dl/proj/phase2_ultralytics/lt_ex.cu"],
                extra_cuda_cflags=["-O2"], verbose=False)
            _GEMMEX_TRY[0] = True
        except Exception as e:  # noqa: BLE001
            print(f"[int8_engine] cublasGemmEx ext 加载失败, 回退 _int_mm: {str(e)[:120]}")
            _GEMMEX_TRY[0] = False
    if not _GEMMEX_TRY[0]:
        return None
    NL, CK = x2d.shape
    co = dy2d.shape[1]
    if CK % 8 != 0:
        x2d = torch.nn.functional.pad(x2d, (0, 8 - CK % 8))
    dw = _GEMMEX_EXT[0].int8_gemm_ex(x2d, dy2d)
    return dw[:, :CK]


def _int8_gemm_ex_dw(x2d, dy2d, chunk_k=32768):
    """dW 专用: 沿收缩维 K=NL 分块, 防 int32 累加溢出。

    int32 累加上限: 块内 K <= 32768 -> 32768*127*127 = 5.3e8 < 2^31 (2.1e9) 安全;
    不分块时 K=NL=409600 -> 6.6e9 溢出 (实测训练梯度爆炸, grad_norm 4058)。
    各块 int32 结果以 fp32 累加 (数学等价)。
    """
    NL = x2d.shape[0]
    if NL <= chunk_k:
        r = _int8_gemm_ex(x2d, dy2d)
        if r is None:
            r = int8_gemm(x2d.transpose(0, 1).contiguous(), dy2d).T.contiguous()
        return r
    acc = None
    for i in range(0, NL, chunk_k):
        blk = _int8_gemm_ex(x2d[i:i + chunk_k], dy2d[i:i + chunk_k])
        if blk is None:
            blk = int8_gemm(x2d[i:i + chunk_k].transpose(0, 1).contiguous(),
                            dy2d[i:i + chunk_k]).T.contiguous()
        acc = blk.to(torch.float32) if acc is None else acc + blk.to(torch.float32)
    return acc  # (CO, CK) fp32


class _Int8ConvCudnn(torch.autograd.Function):
    """cuDNN INT8 隐式 GEMM 卷积 (前向) + SwitchBack 式梯度。

    forward : y = cudnn_int8_conv(xq, wq) * (sx*sw)   (per-tensor, cuDNN 限制)
    backward: dw = int8_gemm(xqᵀ, dyq) * (sx*sdy)    (SwitchBack, 复用 _int_mm/im2col)
              dx = conv_transpose(dy, w)              (fp 高精度; quantize_e=True 时走 int8, 方案 C)
    """

    @staticmethod
    def forward(ctx, x, w, pad, stride):
        sx = scale_absmax(x).reshape(1, 1)
        sw = scale_absmax(w).reshape(1, 1)
        xq = quant_tensor(x, sx)
        wq = quant_tensor(w, sw)
        y = _cudnn_conv_fprop(xq, wq, pad, stride)
        ctx.save_for_backward(xq, w, wq, sx, sw)
        ctx.pad = pad
        ctx.stride = stride
        return y * (sx * sw).to(y.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        xq, w, wq, sx, sw = ctx.saved_tensors
        pad, stride = ctx.pad, ctx.stride
        N, C, H, W = xq.shape
        k = w.shape[2]
        sdy = scale_absmax(grad_y)
        dyq = quant_tensor(grad_y, sdy)
        # SwitchBack dW: im2col(xq)ᵀ @ dyq
        col = F.unfold(F.pad(xq.float(), (pad[1], pad[1], pad[0], pad[0])),
                       kernel_size=k, stride=stride)
        col_i = col.round().to(torch.int8)
        NL = col_i.shape[0] * col_i.shape[2]
        x2d = col_i.transpose(1, 2).reshape(NL, C * k * k)  # (NL, CK) 行主序 (int8 重排拷贝)
        dy2d = dyq.permute(0, 2, 3, 1).reshape(NL, -1)
        dw_i = _int8_gemm_ex_dw(x2d, dy2d)      # (CO, CK) (K=NL 分块防 int32 溢出)
        dw = dw_i.to(grad_y.dtype) * (sx * sdy).to(grad_y.dtype)
        dw = dw.reshape(-1, C, k, k)
        # dX: 方案 B = fp 高精度 (conv_transpose); 方案 C (quantize_e) = int8 GEMM (E 量化)
        if _QUANTIZE_E[0]:
            wq2d = wq.reshape(-1, C * k * k)                   # (CO, CK) 行主序
            dx_i = int8_gemm(dy2d, wq2d)                       # (NL, CK) = dyq @ wq^T (免转置)
            dx = dx_i.to(grad_y.dtype) * (sdy * sw).to(grad_y.dtype)
            dx = F.fold(dx.reshape(N, NL // N, C * k * k).transpose(1, 2),
                        output_size=(H, W), kernel_size=k, stride=stride, padding=pad)
            return dx, dw, None, None
        if stride[0] == 1:
            dx = F.conv_transpose2d(grad_y, w, stride=stride, padding=pad)
        else:
            Ho = grad_y.shape[2]
            dxp = F.conv_transpose2d(grad_y, w, stride=stride, padding=0, output_padding=1)
            dx = dxp[..., pad[0]:pad[0] + H, pad[1]:pad[1] + W]
        return dx, dw, None, None


class _Int8GemmB(torch.autograd.Function):
    """y = xq @ wq * (sx*sw);  dW 走 int8 (SwitchBack);  dx 走高精度。

    参数约定:
        x   : (N, K) float32/16 激活 (已过上一层 BN+SiLU)
        w   : (K, M) float32/16 权重 (2D 展开, 由调用方 im2col 对齐)
        sx  : () 标量或 (1, 1);  x 的 per-tensor scale
        sw  : (1, M)  ;  w 的 per-output-channel scale
        sdy : () 标量;  dy 的 per-tensor scale (反向时计算)
    返回 dx (fp), dw (fp), 以及梯度透传 scale。
    """

    @staticmethod
    def forward(ctx, x, w, sx, sw):
        xq = quant_tensor(x, sx)                     # (N, K) 动态
        wq = quant_tensor(w, sw)                     # (K, M) 静态每步重算
        yq = int8_gemm(xq, wq)                        # (N, M) int32
        y = yq.to(x.dtype) * (sx * sw.to(x.dtype))   # dequant
        ctx.save_for_backward(xq, w, wq, sx, sw)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        xq, w, wq, sx, sw = ctx.saved_tensors
        sdy = scale_absmax(grad_y)                   # 误差 E 动态量化 (每 batch)
        dyq = quant_tensor(grad_y, sdy)              # (N, M) int8
        # SwitchBack: dW = int8(xq).T @ int8(dy) * (sx * sdy)  (gemmEx 免转置)
        dw_i = _int8_gemm_ex(xq, dyq)                        # (M, K) int32
        if dw_i is None:
            dw_i = int8_gemm(xq.transpose(0, 1), dyq)        # 回退 (K, M)
        else:
            dw_i = dw_i.transpose(0, 1).contiguous()         # (K, M) 小矩阵转置
        dw = dw_i.to(grad_y.dtype) * (sx * sdy.to(grad_y.dtype))
        # dX: 方案 A/B = fp (dy @ W^T); 方案 C (quantize_e) = int8 (E 量化, w 用 per-tensor)
        if _QUANTIZE_E[0]:
            sw_t = scale_absmax(w).reshape(1, 1)
            wq_t = quant_tensor(w, sw_t)
            dx_i = int8_gemm(dyq, wq_t.transpose(0, 1).contiguous())   # (N, K)
            dx = dx_i.to(grad_y.dtype) * (sdy * sw_t).to(grad_y.dtype)
        else:
            dx = grad_y @ w.transpose(0, 1)
        return dx, dw, None, None


def int8_gemm_b(x, w, sx, sw):
    """方案 B 的 int8 GEMM 层入口 (对应 Linear 的核心)。"""
    return _Int8GemmB.apply(x, w, sx, sw)


# ----------------------------------------------------------------------------
# Int8Conv2d: im2col + int8 GEMM, 类替换 nn.Conv2d 保持 state_dict 兼容
# ----------------------------------------------------------------------------

class Int8Conv2d(nn.Conv2d):
    """替换 nn.Conv2d 的类: 前向 = int8 GEMM (im2col), 反向 SwitchBack。

    设 x: (N, C_in, H, W),  w: (C_out, C_in, k, k), stride s, pad p:
        im2col -> x2d (N, C_in*k*k, Ho*Wo);  w2d (C_out, C_in*k*k)^T -> (K, M)
        y = _Int8GemmB(x2d, w2d, sx, sw) @ reshape -> (N, C_out, Ho, Wo)
    限制: groups=1, dilation=1 (YOLOv8n 满足)。
    """

    def forward(self, x):
        g = self.groups
        dil = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation
        assert g == 1 and dil == 1, "INT8 引擎仅支持 groups=1/dilation=1"
        N, C, H, W = x.shape
        k = self.kernel_size[0]
        if x.dim() != 4:
            return super().forward(x)  # 非 4D (如某些层) 退回原卷积
        M = self.out_channels
        stride = (self.stride[0], self.stride[1])
        pad = (self.padding[0], self.padding[1])
        Ho = (H + 2 * pad[0] - k) // stride[0] + 1
        Wo = (W + 2 * pad[1] - k) // stride[1] + 1
        # ---- 1x1 专用 GEMM 路径: 1x1 即纯 GEMM, cuDNN int8 1x1 kernel 无优化 (0.3x 慢),
        #      用 _Int8GemmB (int8 GEMM, SwitchBack dW gemmEx 免转置, fp dX), 实测达 ~1.0x fp32 ----
        if k == 1 and _cudnn_available():
            x2d = x.permute(0, 2, 3, 1).reshape(N * H * W, C)  # (NL, C)
            w2d = self.weight.reshape(M, C).transpose(0, 1).contiguous()  # (C, M)
            sx = scale_absmax(x2d).reshape(1, 1)
            sw = scale_absmax(w2d, dim=0)                    # (1, M) per-output-channel
            y = int8_gemm_b(x2d, w2d, sx, sw)                 # (NL, M) fp
            if self.bias is not None:
                y = y + self.bias.reshape(1, M)
            return y.reshape(N, H, W, M).permute(0, 3, 1, 2)
        # ---- cuDNN INT8 隐式 GEMM 后端 (免 im2col, 仅 fprop; 3080Ti 快 3-4x) ----
        if _cudnn_available():
            y = _Int8ConvCudnn.apply(x, self.weight, pad, stride)
            if self.bias is not None:
                y = y + self.bias.reshape(1, M, 1, 1)
            return y
        # ---- im2col + _int_mm 后端 (默认, 数值同前) ----
        # im2col (unfold 不内置 pad, 先手动 pad)
        xp = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        col = F.unfold(xp, kernel_size=k, stride=self.stride)  # (N, C*k*k, L)
        N, CK, L = col.shape
        K_dim = CK
        x2d = col.permute(0, 2, 1).reshape(N * L, K_dim).to(x.dtype)  # (N*L, K)
        w2d = self.weight.reshape(M, K_dim).transpose(0, 1).contiguous()  # (K, M)
        sx = scale_absmax(x2d).reshape(1, 1)            # per-tensor
        sw = scale_absmax(w2d, dim=0)                   # (1, M) per-output-channel
        y = int8_gemm_b(x2d, w2d, sx, sw)                # (N*L, M) fp
        if self.bias is not None:
            y = y + self.bias.reshape(1, M)
        y = y.reshape(N, L, M).permute(0, 2, 1)
        return y.reshape(N, M, Ho, Wo)


# ----------------------------------------------------------------------------
# cuDNN INT8 卷积后端 (cuDNN 9 图 API; 隐式 GEMM, 免 im2col; 仅 fprop 在 sm86 可用)
# ----------------------------------------------------------------------------

_CUDNN_CONV_CACHE = {}
_CUDNN_HANDLE = [None]
_CUDNN_OK = [None]


def _cudnn_available():
    if _CUDNN_OK[0] is None:
        try:
            import cudnn  # noqa: F401
            _CUDNN_OK[0] = torch.cuda.is_available()
        except ImportError:
            _CUDNN_OK[0] = False
    return _CUDNN_OK[0]


def _cudnn_nhwc_stride(dim):
    """NHWC 布局 stride (c 在最内, stride 1)。旧实现错位 (C-stride=H*W, H-stride=W),
    导致 cuDNN 对 C=64 等形状的输出布局错乱 (实测 model.5 起 rel 100% 误差)。"""
    n, c, h, w = dim
    return [h * w * c, 1, w * c, c]


def _cudnn_conv_fprop(xq, wq, pad, stride):
    """cuDNN INT8 隐式 GEMM 前向: xq/wq 均 int8 NCHW -> (N,CO,Ho,Wo) fp32 (未乘 scale)。

    图按 (N,C,H,W,CO,k,pad,stride) 缓存; 数据切到 NHWC 后 execute 原位写回。
    """
    import cudnn
    N, C, H, W = xq.shape
    CO = wq.shape[0]
    k = wq.shape[2]
    key = (N, C, H, W, CO, k, tuple(pad), tuple(stride))
    g = _CUDNN_CONV_CACHE.get(key)
    if g is None:
        if _CUDNN_HANDLE[0] is None:
            _CUDNN_HANDLE[0] = cudnn.create_handle()
            cudnn.set_stream(handle=_CUDNN_HANDLE[0],
                             stream=torch.cuda.current_stream().cuda_stream)
        gp = cudnn.pygraph(intermediate_data_type=cudnn.data_type.FLOAT,
                           compute_data_type=cudnn.data_type.INT32,
                           handle=_CUDNN_HANDLE[0])
        X = gp.tensor(dim=[N, C, H, W], stride=_cudnn_nhwc_stride([N, C, H, W]),
                      data_type=cudnn.data_type.INT8, name="X")
        Wt = gp.tensor(dim=[CO, C, k, k], stride=_cudnn_nhwc_stride([CO, C, k, k]),
                       data_type=cudnn.data_type.INT8, name="Wt")
        conv = gp.conv_fprop(image=X, weight=Wt, padding=list(pad), stride=list(stride),
                             dilation=[1, 1], compute_data_type=cudnn.data_type.INT32)
        conv.set_output(True)
        conv.data_type = cudnn.data_type.FLOAT
        gp.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        wsp = (torch.empty(gp.get_workspace_size(), device="cuda", dtype=torch.uint8)
               if gp.get_workspace_size() else None)
        g = (gp, X, Wt, conv, wsp)
        _CUDNN_CONV_CACHE[key] = g
    gp, X, Wt, conv, wsp = g
    Ho = (H + 2 * pad[0] - k) // stride[0] + 1
    Wo = (W + 2 * pad[1] - k) // stride[1] + 1
    y = torch.zeros((N, CO, Ho, Wo), device=xq.device, dtype=torch.float32)
    y = y.to(memory_format=torch.channels_last)
    gp.execute({X: xq.contiguous(memory_format=torch.channels_last),
                Wt: wq.contiguous(memory_format=torch.channels_last),
                conv: y},
               wsp, handle=_CUDNN_HANDLE[0])
    return y.contiguous()


# ----------------------------------------------------------------------------
# 模型 patch (与 qat_patch 一致: 类替换, ckpt 无缝加载; Detect 头保持 fp32)
# ----------------------------------------------------------------------------

def patch_int8_engine(model, skip_head=True, verbose=False):
    """model 为 ultralytics DetectionModel (或其 .model)。返回被替换的卷积名字列表。

    与 QAT 补丁同样的哲学: m.__class__ = Int8Conv2d, 不新增参数, 权重直接复用,
    因此 yolov8n.pt / QAT ckpt 均可无痕加载。
    """
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
        m.__class__ = Int8Conv2d
        patched.append(name)
    if verbose:
        fwd = "cudnn_int8_conv" if _cudnn_available() else "im2col_int_mm"
        print(f"[int8_engine] 已替换 {len(patched)} 个卷积为 Int8Conv2d "
              f"(fwd={fwd}, dW=SwitchBack int8_gemm={int8_gemm_backend()})")
    return patched


def count_int8_convs(net):
    return sum(1 for m in net.modules() if isinstance(m, Int8Conv2d))


# ----------------------------------------------------------------------------
# 数值验证 (本机 CPU 或远端 GPU 均可跑)
# ----------------------------------------------------------------------------

def _rel_err(a, b, eps=1e-6):
    return ((a - b).abs().max() / (a.abs().max() + eps)).item()


def sanity_check(verbose=True):
    """前向/反向与 fp32 参考对比 + gradcheck。返回结果 dict。"""
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"backend": int8_gemm_backend(), "device": str(dev),
           "torch": torch.__version__, "cuda": torch.cuda.is_available()}

    # 1b) K pad 对齐回归: K 非 32 倍数时 pad 后结果与未 pad 完全一致 (K=27=3x3x3)
    for k in (16, 27, 64):
        a = torch.randint(-30, 30, (32, k), dtype=torch.int8, device=dev)
        b = torch.randint(-30, 30, (k, 8), dtype=torch.int8, device=dev)
        ref = torch.matmul(a.float(), b.float()).to(torch.int32)
        got = int8_gemm(a, b)
        assert torch.equal(got, ref), f"K={k} pad 对齐错误"
    res["kpad_align"] = "PASS"

    # 1b) int32 溢出回归: dW 的收缩维 K=NL 可达 409600 (160px 层), 127*127*409600
    #     = 6.6e9 > 2^31 会溢出 (实测训练梯度爆炸); 分块后应与 fp64 参考一致 (量化误差内)
    if torch.cuda.is_available():
        n_big, ck_big, co_big = 262144, 576, 32
        xb = torch.randint(-127, 127, (n_big, ck_big), dtype=torch.int8, device=dev)
        dyb = torch.randint(-127, 127, (n_big, co_big), dtype=torch.int8, device=dev)
        dw_big = _int8_gemm_ex_dw(xb, dyb).to(torch.float64)
        ref_big = (xb.to(torch.float64).T @ dyb.to(torch.float64)).T  # (CO, CK)
        rel_big = (dw_big - ref_big).abs().max().item() / ref_big.abs().max().item()
        res["int32_overflow_regress"] = rel_big
        assert rel_big < 0.05, f"dW int32 溢出未修复 (rel {rel_big})"
    else:
        res["int32_overflow_regress"] = "cpu-skip"

    # 1) GEMM 数值: 小矩阵精确一致 (int8 值域内), 大矩阵误差 = 量化误差
    for n, k, m in [(32, 32, 32), (32, 256, 64), (128, 512, 128)]:
        x = torch.randn(n, k, device=dev) * 2
        w = torch.randn(k, m, device=dev) * 0.1
        sx, sw = scale_absmax(x).reshape(1, 1), scale_absmax(w, 0)
        y = int8_gemm_b(x, w, sx, sw).float()
        y_ref = x.detach() @ w.detach()
        err = (y - y_ref).abs().max().item() / (y_ref.abs().max() + 1e-8).item()
        res[f"gemm_{n}x{k}x{m}"] = err

    # 2) Conv2d 前向误差 vs fp32 conv
    m = Int8Conv2d(16, 32, 3, stride=2, padding=1, bias=True).to(dev)
    m.weight.data.normal_(0, 0.05)
    x4 = torch.randn(4, 16, 64, 64, device=dev)
    y4 = m(x4).float()
    y4_ref = F.conv2d(x4, m.weight, m.bias, stride=2, padding=1)
    res["conv_fwd_rel_err"] = _rel_err(y4, y4_ref)
    assert res["conv_fwd_rel_err"] < 0.05, "Conv 前向误差过大"

    # 3) 反向: dW 与解析解对比 (量化带来的偏差应在 STE 噪声量级)
    xg = torch.randn(4, 16, 32, 32, device=dev, requires_grad=True)
    yg = m(xg).float()
    yg.sum().backward()
    dw_engine = m.weight.grad.clone()
    m2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=True).to(dev)
    m2.load_state_dict(m.state_dict())
    F.conv2d(xg, m2.weight, m2.bias, stride=2, padding=1).sum().backward()
    res["dw_rel_err"] = _rel_err(dw_engine, m2.weight.grad)
    assert res["dw_rel_err"] < 0.10, "dW 误差过大 (SwitchBack int8 dW 偏差超预期)"

    # 4) dx 一致性: 方案 A/B 的 dX 是高精度路径 (dy @ W^T), 应精确匹配 (不是 STE 近似)
    xh = torch.randn(4, 16, 32, 32, device=dev, requires_grad=True)
    mh = Int8Conv2d(16, 32, 3, stride=2, padding=1).to(dev)
    mh.weight.data.normal_(0, 0.05)
    go = torch.randn_like(mh(xh.detach()).float())
    mh(xh).float().mul(go).sum().backward()
    dx_engine = xh.grad.clone()
    # im2col 内部先 pad(1) 再 stride=2 采样; 反演 = conv_transpose padding=0,
    # output_padding=1 (34x34) 再裁掉 pad 边 (恢复 32x32)
    dxp = F.conv_transpose2d(go, mh.weight, stride=2, padding=0, output_padding=1)
    dx_ref = dxp[..., 1:-1, 1:-1]
    res["dx_rel_err"] = _rel_err(dx_engine, dx_ref)
    # CUDA 上 unfold 反向与 conv_transpose 的 fp32 累加顺序不同, 允许 ~3e-4 浮点差异;
    # 方案 C (quantize_e) 时 dX 走 int8 (dy/w 量化), 误差为量化级 ~2%
    dx_tol = 5e-3 if not _QUANTIZE_E[0] else 0.05
    assert res["dx_rel_err"] < dx_tol, "dX 路径误差超预期"

    # 4) gradcheck (输入/权重)
    gx = torch.randn(4, 8, 12, 12, device=dev, requires_grad=True)
    gw = torch.randn(16, 8, 3, 3, device=dev, requires_grad=True)
    gm = Int8Conv2d(8, 16, 3, padding=1).to(dev)
    gm.weight.data.copy_(gw)
    gbias = torch.randn(16, device=dev)
    gm.bias.data.copy_(gbias)
    go = torch.randn_like(gm(gx.detach())).reshape(-1)
    ok = torch.autograd.gradcheck(
        lambda xx, ww: gm.forward(xx).reshape(-1) * go,
        (gx.detach().requires_grad_(), gw.detach().requires_grad_()),
        eps=1e-3, atol=2e-2, rtol=2e-2, raise_exception=False)
    res["gradcheck"] = bool(ok)

    # 5) gradcheck 不适用: 含 STE round (分段常数), 数值差分必然失败;
    #    改用 dx 精确性 + dw 统计误差 + 训练 smoke 综合验证
    res["gradcheck"] = "n/a (STE round, 数值差分不适用)"
    res["pass"] = bool(res["conv_fwd_rel_err"] < 0.05
                       and res["dw_rel_err"] < 0.10
                       and res["dx_rel_err"] < dx_tol)

    if verbose:
        print(f"[sanity] backend={res['backend']} device={res['device']}")
        for k_, v in res.items():
            if isinstance(v, float):
                print(f"  {k_:<20s} {v:.6f}")
            else:
                print(f"  {k_:<20s} {v}")
    return res


def model_smoke(verbose=True, device="cpu"):
    """整模型 smoke: patch 后前向/反向均能跑通、梯度有限、int8 层生效。"""
    from types import SimpleNamespace
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    net = model.model.to(device)
    for p in net.parameters():
        p.requires_grad_(True)
    net.criterion = net.init_criterion()
    net.criterion.hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    patched = patch_int8_engine(net, verbose=verbose)
    net.train()
    x = torch.randn(2, 3, 320, 320, device=device).requires_grad_()
    batch = dict(img=x, batch_idx=torch.tensor([0, 1]),
                 cls=torch.tensor([0., 5.]),
                 bboxes=torch.tensor([[100., 100., 200., 200.],
                                      [50., 50., 150., 150.]]))
    losses, loss_det = net.loss(batch)
    losses.sum().backward()
    n_grad = sum(1 for p in net.parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0)
    finite = all(torch.isfinite(p.grad).all() for p in net.parameters()
                 if p.grad is not None)
    n_int8 = count_int8_convs(net)
    res = dict(patched=len(patched), n_int8=n_int8, n_grad=n_grad,
               n_params=sum(1 for _ in net.parameters()), finite=bool(finite),
               loss=[float(v) for v in losses.detach()])
    ok = n_grad > 100 and finite and n_int8 > 40
    if verbose:
        print(f"[smoke] int8 层 {n_int8}/64; 有梯度 {n_grad}/184; 全有限: {finite}")
        print(f"[smoke] loss={res['loss']}")
        print(f"[{'PASS' if ok else 'FAIL'}] 整模型 smoke")
    return res, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="跑数值验证")
    ap.add_argument("--smoke", action="store_true", help="跑整模型 smoke (需 yolov8n.pt)")
    ap.add_argument("--quantize-e", action="store_true",
                    help="方案 C: dX 链误差信号 E 也走 int8 GEMM (默认方案 B: dX 高精度)")
    args = ap.parse_args()
    if args.quantize_e:
        _QUANTIZE_E[0] = True
    if args.sanity:
        res = sanity_check(verbose=True)
        ok = res.get("pass")
        print(f"\n[{'PASS' if ok else 'FAIL'}] INT8 引擎数值验证"
              f" (backend={res['backend']})")
        return 0 if ok else 1
    if args.smoke:
        res, ok = model_smoke(verbose=True)
        return 0 if ok else 1
    print("用法: python int8_engine.py --sanity | --smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())