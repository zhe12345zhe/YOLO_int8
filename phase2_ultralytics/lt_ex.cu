#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")

static cublasHandle_t g_h = nullptr;

// x: (nl, ck) int8 行主序; dy: (nl, co) int8 行主序 -> (co, ck) int32 行主序
torch::Tensor int8_gemm_ex(torch::Tensor x, torch::Tensor dy) {
    CHECK_CUDA(x); CHECK_CUDA(dy);
    TORCH_CHECK(x.is_contiguous() && dy.is_contiguous(), "contiguous required");
    const int64_t nl = x.size(0), ck = x.size(1);
    const int64_t co = dy.size(1);
    TORCH_CHECK(dy.size(0) == nl, "K mismatch");
    if (!g_h) cublasCreate(&g_h);
    auto outc = torch::empty({co, ck}, x.options().dtype(torch::kInt));
    // 列主序: A (m=ck, k=nl) ld=ck [transa=N, 存储 torch (nl,ck) 行主序]
    //         B (n=co, k=nl) ld=co [transb=T, 存储 torch (nl,co) 行主序]
    //         C (m=ck, n=co) ld=ck [torch (co,ck) 行主序]
    const int32_t alpha = 1, beta = 0;
    cublasStatus_t st = cublasGemmEx(g_h,
        CUBLAS_OP_N, CUBLAS_OP_T,
        (int)ck, (int)co, (int)nl,
        &alpha, (const void*)x.data_ptr(), CUDA_R_8I, (int)ck,
                (const void*)dy.data_ptr(), CUDA_R_8I, (int)co,
        &beta, (void*)outc.data_ptr(), CUDA_R_32I, (int)ck,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasGemmEx failed code=", (int)st);
    return outc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm_ex", &int8_gemm_ex, "cublasGemmEx int8 with transpose");
}
