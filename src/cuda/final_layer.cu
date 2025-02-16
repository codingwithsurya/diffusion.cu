// final_layer.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cublas_v2.h>

// A simple device function for SiLU
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// A minimal kernel to apply SiLU elementwise
__global__ void silu_kernel(const float* __restrict__ inp,
                            float* __restrict__ out,
                            int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = inp[idx];
        out[idx] = silu(val);
    }
}

// A small kernel to add bias
__global__ void add_bias_kernel(float* __restrict__ data,
                                const float* __restrict__ bias,
                                int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx % dim;
        data[idx] += bias[c];
    }
}

// final_layer_forward:
//   x: [B, T, C]
//   c: [B, C]
//   w_mod, b_mod: shapes [C, 2*C] and [2*C]
//   w_linear, b_linear: shapes [C, out_dim]
// Returns: [B, T, out_dim]
torch::Tensor final_layer_forward(
    torch::Tensor x,        // [B, T, C]
    torch::Tensor c,        // [B, C]
    torch::Tensor w_mod,    // [C, 2*C]
    torch::Tensor b_mod,    // [2*C]
    torch::Tensor w_linear, // [C, out_dim]
    torch::Tensor b_linear, // [out_dim]
    float eps = 1e-6
) {
    TORCH_CHECK(x.is_cuda() && c.is_cuda(), "x and c must be CUDA");
    TORCH_CHECK(w_mod.is_cuda() && b_mod.is_cuda(), "w_mod,b_mod must be CUDA");
    TORCH_CHECK(w_linear.is_cuda() && b_linear.is_cuda(), "w_linear,b_linear must be CUDA");

    int B = x.size(0);
    int T = x.size(1);
    int C_in = x.size(2);
    int out_dim = w_linear.size(1);

    auto opts = x.options();
    // 1) c_silu = SiLU(c) => shape [B, C]
    auto c_silu = torch::empty_like(c);
    {
        int total = B * C_in;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        silu_kernel<<<grid, block>>>(c.data_ptr<float>(), c_silu.data_ptr<float>(), total);
        cudaDeviceSynchronize();
    }

    // 2) mod_out = c_silu @ w_mod + b_mod => shape [B, 2*C]
    cublasHandle_t handle;
    cublasCreate(&handle);

    auto mod_out = torch::empty({B, 2 * C_in}, opts);
    {
        float alpha = 1.f;
        float beta = 0.f;
        // c_silu: [B, C], w_mod: [C, 2*C] => out: [B, 2*C]
        cublasSgemm(
            handle,
            CUBLAS_OP_N,      // w_mod not transposed
            CUBLAS_OP_N,      // c_silu not transposed
            2*C_in,           // m
            B,                // n
            C_in,             // k
            &alpha,
            w_mod.data_ptr<float>(), 2*C_in, // leading dim
            c_silu.data_ptr<float>(), C_in,
            &beta,
            mod_out.data_ptr<float>(), 2*C_in
        );
        // Add b_mod
        int total = B * (2*C_in);
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block>>>(
            mod_out.data_ptr<float>(),
            b_mod.data_ptr<float>(),
            total,
            2*C_in
        );
        cudaDeviceSynchronize();
    }

    cublasDestroy(handle);

    // 3) chunk(2, dim=1) => shift, scale => each [B, C]
    auto chunked = mod_out.chunk(2, 1);
    auto shift = chunked[0]; // [B, C]
    auto scale = chunked[1]; // [B, C]

    // 4) LN x => x_ln
    //    we do a simple per-(b,t) LN: mean & var over dim C
    auto x_ln = torch::empty_like(x);
    {
        // For each (b, t), compute mean,var => normalize => store
        // We can do it in a naive kernel or in a single pass on CPU. We'll do it on GPU quickly:
        // Not heavily optimized, but good for demonstration.
        const int total = B * T;
        const int D = C_in;

        // We'll do a simple kernel:
        static __global__ void ln_kernel(
            const float* __restrict__ inp,
            float* __restrict__ out,
            int B, int T, int C, float eps
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= B*T) return;
            int b_t = idx;
            int b_ = b_t / T;
            int t_ = b_t % T;
            const float* row_in = inp + ((long long) b_ * T + t_) * C;
            float* row_out = out + ((long long) b_ * T + t_) * C;

            // compute mean
            float mean = 0.f;
            for (int i = 0; i < C; i++) {
                mean += row_in[i];
            }
            mean /= C;
            // compute var
            float var = 0.f;
            for (int i = 0; i < C; i++) {
                float diff = row_in[i] - mean;
                var += diff * diff;
            }
            var /= C;
            float inv_std = rsqrtf(var + eps);

            // write out
            for (int i = 0; i < C; i++) {
                row_out[i] = (row_in[i] - mean) * inv_std;
            }
        };

        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        ln_kernel<<<grid, block>>>(x.data_ptr<float>(), x_ln.data_ptr<float>(), B, T, C_in, eps);
        cudaDeviceSynchronize();
    }

    // 5) x_mod = scale * x_ln + shift => shape [B, T, C]
    //    scale, shift: [B, C], need to broadcast over T
    //    We'll do a small kernel:
    {
        static __global__ void mod_kernel(
            const float* __restrict__ x_ln,
            const float* __restrict__ shift,
            const float* __restrict__ scale,
            float* __restrict__ x_mod,
            int B, int T, int C
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= B * T * C) return;
            int c = idx % C;
            int b_t = idx / C;
            int b = b_t / T;
            // shift[b, c], scale[b, c]
            float val = x_ln[idx];
            val = val * scale[b*C + c] + shift[b*C + c];
            x_mod[idx] = val;
        };

        int total2 = B * T * C_in;
        dim3 block(256);
        dim3 grid((total2 + block.x - 1) / block.x);
        mod_kernel<<<grid, block>>>(
            x_ln.data_ptr<float>(),
            shift.data_ptr<float>(),
            scale.data_ptr<float>(),
            x_ln.data_ptr<float>(), // do it in-place if you want
            B, T, C_in
        );
        cudaDeviceSynchronize();
    }

    // 6) final linear => out => shape [B, T, out_dim]
    auto x_flat = x_ln.reshape({B * T, C_in});
    auto out = torch::empty({B*T, out_dim}, opts);

    {
        cublasHandle_t handle2;
        cublasCreate(&handle2);
        float alpha = 1.f;
        float beta = 0.f;
        // x_flat: [B*T, C], w_linear: [C, out_dim], => out: [B*T, out_dim]
        cublasSgemm(
            handle2,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            out_dim,
            B*T,
            C_in,
            &alpha,
            w_linear.data_ptr<float>(), out_dim,
            x_flat.data_ptr<float>(), C_in,
            &beta,
            out.data_ptr<float>(), out_dim
        );
        // add bias
        int total = (B*T) * out_dim;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block>>>(
            out.data_ptr<float>(),
            b_linear.data_ptr<float>(),
            total,
            out_dim
        );
        cudaDeviceSynchronize();

        cublasDestroy(handle2);
    }

    // reshape final to [B, T, out_dim]
    return out.reshape({B, T, out_dim});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("final_layer_forward", &final_layer_forward, "Final Layer forward (CUDA)");
}