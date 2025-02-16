// cuda_util_kernels.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Tiling parameters for modulate kernel.
#define TILE_T 32
#define TILE_D 32

//-------------------------------------------------------------------------
// 1. Modulate Kernel
// Computes for each sample (n):
//   out[n, t, d] = x[n, t, d] * (1 + scale[n, d]) + shift[n, d]
// x: [N, T, D], shift/scale: [N, D]
__global__ void modulate_kernel(const float* __restrict__ x,
                                const float* __restrict__ shift,
                                const float* __restrict__ scale,
                                float* __restrict__ out,
                                int T, int D)
{
    int n = blockIdx.z;           // sample index
    int t_start = blockIdx.x * TILE_T;
    int d_start = blockIdx.y * TILE_D;
    int t_idx = threadIdx.x;
    int d_idx = threadIdx.y;
    int t = t_start + t_idx;
    int d = d_start + d_idx;

    // Load per-sample shift/scale into shared memory
    __shared__ float s_shared[TILE_D];
    __shared__ float sh_shared[TILE_D];
    for (int i = d_idx; i < TILE_D; i += blockDim.y) {
        int d_global = d_start + i;
        if (d_global < D) {
            s_shared[i] = scale[n * D + d_global];
            sh_shared[i] = shift[n * D + d_global];
        }
    }
    __syncthreads();

    if (t < T && d < D) {
        int idx = n * T * D + t * D + d;
        float s_val = s_shared[d_idx];
        float sh_val = sh_shared[d_idx];
        out[idx] = x[idx] * (1.0f + s_val) + sh_val;
    }
}

//-------------------------------------------------------------------------
// 2. SinCos Positional Embedding Kernel
// Computes a 2D sincos positional embedding for a grid of size (grid_size x grid_size).
// The output tensor has shape: [grid_size*grid_size, embed_dim].
// We split embed_dim into two halves (for y and x) and, for each half,
// compute sin and cos values using frequencies based on 10000.
__global__ void sincos_pos_embed_kernel(float* __restrict__ pos_embed,
                                          int embed_dim, int grid_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_positions = grid_size * grid_size;
    if (idx < num_positions) {
        int i = idx / grid_size;  // row (y coordinate)
        int j = idx % grid_size;  // column (x coordinate)
        int D = embed_dim;
        int half = D / 2;
        int quarter = half / 2;
        float* out_ptr = pos_embed + idx * D;
        // For the y coordinate:
        for (int k = 0; k < quarter; k++) {
            float omega = expf(-logf(10000.0f) * ((float)k / (float)half));
            float val = i * omega;
            out_ptr[k] = sinf(val);
            out_ptr[k + quarter] = cosf(val);
        }
        // For the x coordinate:
        for (int k = 0; k < quarter; k++) {
            float omega = expf(-logf(10000.0f) * ((float)k / (float)half));
            float val = j * omega;
            out_ptr[half + k] = sinf(val);
            out_ptr[half + k + quarter] = cosf(val);
        }
    }
}

//-------------------------------------------------------------------------
// 3. Wrappers to be called from Python
// These functions take PyTorch tensors as input, launch the CUDA kernels,
// and return output tensors.

// modulate_forward:
//   x: [N, T, D], shift/scale: [N, D]
//   Returns: [N, T, D]
torch::Tensor modulate_forward(torch::Tensor x, torch::Tensor shift, torch::Tensor scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(shift.is_cuda(), "shift must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

    int N = x.size(0);
    int T = x.size(1);
    int D = x.size(2);
    auto out = torch::empty_like(x);
    dim3 blockDim(TILE_T, TILE_D, 1);
    dim3 gridDim((T + TILE_T - 1) / TILE_T,
                 (D + TILE_D - 1) / TILE_D,
                 N);
    modulate_kernel<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        shift.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        T, D);
    cudaDeviceSynchronize();
    return out;
}

// sincos_pos_embed_forward:
//   Given embed_dim and grid_size, returns a tensor of shape [grid_size*grid_size, embed_dim]
torch::Tensor sincos_pos_embed_forward(int embed_dim, int grid_size) {
    int num_positions = grid_size * grid_size;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto pos_embed = torch::empty({num_positions, embed_dim}, options);
    int threadsPerBlock = 256;
    int blocks = (num_positions + threadsPerBlock - 1) / threadsPerBlock;
    sincos_pos_embed_kernel<<<blocks, threadsPerBlock>>>(
        pos_embed.data_ptr<float>(), embed_dim, grid_size);
    cudaDeviceSynchronize();
    return pos_embed;
}

//-------------------------------------------------------------------------
// 4. Bindings (using pybind11)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("modulate_forward", &modulate_forward, "Modulate forward (CUDA)");
    m.def("sincos_pos_embed_forward", &sincos_pos_embed_forward, "Sincos Pos Embed forward (CUDA)");
}

