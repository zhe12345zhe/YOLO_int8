"""纯 GEMM 对比 (预量化输入, 不含量化扫描): fp32 matmul vs int8 _int_mm/gemmEx。"""
import os, sys, time
sys.path.insert(0, "/root/dl/proj/phase2_ultralytics")
os.environ["PATH"] = "/root/miniconda3/bin:" + os.environ.get("PATH", "")
import torch
import int8_engine as ie

dev = "cuda"

def bench(fn, iters=50, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3

print(f"{'形状':<22} {'fp32 matmul':>12} {'int8 _int_mm':>12} {'int8/fp32':>10} {'int8 加速':>9}")
for name, NL, CK, M in [("160px", 409600, 576, 64),
                        ("80px", 102400, 1152, 128),
                        ("40px", 25600, 2304, 256),
                        ("20px", 6400, 4608, 512),
                        ("160px宽", 409600, 1152, 128)]:
    xf = torch.randn(NL, CK, device=dev)
    wf = torch.randn(CK, M, device=dev)
    t_fp = bench(lambda: xf @ wf)
    # int8: 预量化 (不算量化时间)
    sx = xf.abs().amax() / 127.0
    sw = wf.abs().amax() / 127.0
    xq = (xf / sx).round().clamp(-128, 127).to(torch.int8)
    wq = (wf / sw).round().clamp(-128, 127).to(torch.int8)
    t_i8 = bench(lambda: ie.int8_gemm(xq, wq))
    print(f"{name:<22} {t_fp:12.3f} {t_i8:12.3f} {t_i8/t_fp:10.2f} {t_fp/t_i8:8.2f}x")

# 量化时间占比 (真实引擎里每步的量化开销)
print()
print("=== 量化税 (absmax 扫描 + quant) 单层耗时 ===")
for name, N, C, H, W in [("160px", 16, 64, 160, 160), ("80px", 16, 128, 80, 80)]:
    x = torch.randn(N, C, H, W, device=dev)
    def quant_all():
        s = ie.scale_absmax(x)
        return ie.quant_tensor(x, s)
    t_q = bench(quant_all)
    print(f"  {name}: quant {t_q:.3f} ms (含 absmax+round+cast)")