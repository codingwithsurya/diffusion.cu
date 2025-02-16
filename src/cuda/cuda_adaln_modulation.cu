
// cuda_adaln_modulation.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// A kernel to apply SiLU + bias:  c_out[i] = SiLU( c_in[i] ) * W + b, or we do it in two steps.
__global__ void silu_linear_kernel(
    const float* __restrict__ c_in,
    float* __restrict__ c_out,
    const float* __restrict__ bias,
    int N,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * dim) {
        float val = c_in[idx] + bias[idx % dim];
        val = silu(val);
        c_out[idx] = val;
    }
}

// For large dimension multiplications, we typically use cuBLAS. So we may want cublasSgemm
// to do c_out = SiLU(c_in) * W + B.  The simplest approach is the same pattern as MLP above.

torch::Tensor silu_linear_forward(
    torch::Tensor c,      // [B, d]
    torch::Tensor w,      // [d, out_dim]
    torch::Tensor b       // [out_dim]
) {
    // We'll do: tmp = c + b1 (pointwise), then tmp = silu(tmp), then gemm with w, then + b2 if needed.
    // But from your snippet, it looks like: out = Linear(SiLU(c)), so let's do:
    //    c_silu = SiLU(c)
    //    out = c_silu @ w + b
    // We can do it with a small kernel for the SiLU, then cublasSgemm for the matrix multiply.
    TORCH_CHECK(c.is_cuda() && w.is_cuda() && b.is_cuda(), "All must be CUDA");
    TORCH_CHECK(c.dtype() == torch::kFloat32, "float32 only");
    int B = c.size(0);
    int d = c.size(1);
    int out_dim = w.size(1);

    // 1) c_silu = SiLU(c)
    auto c_silu = torch::empty_like(c);
    {
        int threads = 256;
        int blocks = (B * d + threads - 1) / threads;
        // define a small kernel that does c_silu[i] = silu(c[i])
        static __global__ void silu_kernel(const float* in, float* out, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                float val = in[idx];
                // SiLU
                out[idx] = val / (1.f + expf(-val));
            }
        };
        silu_kernel<<<blocks, threads>>>(c.data_ptr<float>(), c_silu.data_ptr<float>(), B*d);
        cudaDeviceSynchronize();
    }

    // 2) out = c_silu * w + b
    auto out = torch::empty({B, out_dim}, c.options());
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.f;
    float beta = 0.f;
    // shape c_silu: [B, d]
    // shape w: [d, out_dim]
    // -> out: [B, out_dim]
    cublasSgemm(
        handle,
        CUBLAS_OP_N,  // w not transposed
        CUBLAS_OP_N,  // c_silu not transposed
        out_dim,      // m
        B,            // n
        d,            // k
        &alpha,
        w.data_ptr<float>(), out_dim,
        c_silu.data_ptr<float>(), d,
        &beta,
        out.data_ptr<float>(), out_dim
    );

    // Add bias
    {
        int total = B * out_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        static __global__ void add_bias_kernel(float* data, const float* bias, int total, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < total) {
                int c = idx % dim;
                data[idx] += bias[c];
            }
        };
        add_bias_kernel<<<blocks, threads>>>(out.data_ptr<float>(), b.data_ptr<float>(), total, out_dim);
        cudaDeviceSynchronize();
    }

    cublasDestroy(handle);
    return out;
}

// Now for the adaptive layer norm part:
//   x_norm = (x - mean) / sqrt(var + eps)
//   out = scale * x_norm + shift
// We assume shift, scale are [B, C], or [C]? Usually chunk(6,dim=1) => shift is [B,C], scale is [B,C].
__global__ void ada_layernorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    int B,
    int T,
    int C,
    float eps
) {
    // We treat B*T as the batch of vectors of length C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    // offset in x
    int b_t = idx;  // which row among B*T
    // we want to compute mean, var for that row
    const float* x_ptr = x + (long long) b_t * C;
    float mean = 0.f;
    float var = 0.f;

    // 1) compute mean
    for (int i = 0; i < C; i++) {
        mean += x_ptr[i];
    }
    mean /= C;
    // 2) compute var
    for (int i = 0; i < C; i++) {
        float diff = x_ptr[i] - mean;
        var += diff * diff;
    }
    var /= C;

    float inv_std = rsqrtf(var + eps);

    // shift, scale are [B, C], but we need to find which B for shift/scale
    int b_idx = b_t / T;  // integer division
    const float* shift_ptr = shift + (long long) b_idx * C;
    const float* scale_ptr = scale + (long long) b_idx * C;
    float* y_ptr = y + (long long) b_t * C;

    for (int i = 0; i < C; i++) {
        float val = (x_ptr[i] - mean) * inv_std;
        val = val * scale_ptr[i] + shift_ptr[i];
        y_ptr[i] = val;
    }
}

torch::Tensor adaln_forward(
    torch::Tensor x,      // [B, T, C]
    torch::Tensor shift,  // [B, C]
    torch::Tensor scale,  // [B, C]
    float eps
) {
    TORCH_CHECK(x.is_cuda() && shift.is_cuda() && scale.is_cuda(), "Must be CUDA");
    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);

    // output
    auto y = torch::empty_like(x);

    // We launch B*T blocks
    int blocks = B * T;
    int threads = 256;
    int total = blocks;
    int max_blocks = (total + threads - 1) / threads;

    ada_layernorm_kernel<<<max_blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        shift.data_ptr<float>(),
        scale.data_ptr<float>(),
        B,
        T,
        C,
        eps
    );
    cudaDeviceSynchronize();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_linear_forward", &silu_linear_forward, "SiLU + Linear");
    m.def("adaln_forward", &adaln_forward, "Adaptive LN forward (CUDA)");
}


