#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define THREADS 256

// A very simple forward attention kernel.
// Each block computes one output row: out[b, i, :] = sum_j softmax(scores)[j] * V[b, j, :],
// where scores[j] = (Q[b, i, :] dot K[b, j, :]) / sqrt(C).
// For simplicity, only thread 0 of each block computes the weighted sum.
__global__ void attention_forward_kernel(const float* __restrict__ Q,
                                           const float* __restrict__ K,
                                           const float* __restrict__ V,
                                           float* __restrict__ out,
                                           int B, int T, int C) {
    // Grid: (B, T)
    int b = blockIdx.x;  // batch index
    int i = blockIdx.y;  // query (sequence) index
    int tid = threadIdx.x;

    // Shared memory: we'll use T floats to store the dot-product scores.
    extern __shared__ float sdata[];  // size: T * sizeof(float)
    float* scores = sdata; // scores[0..T-1]

    // --- Compute dot-product scores for query Q[b,i,:] with each key K[b,j,:] ---
    // Each thread loops over j in [tid, T, blockDim.x].
    for (int j = tid; j < T; j += blockDim.x) {
        float dot = 0.0f;
        int q_offset = b * T * C + i * C;       // Q[b,i,:]
        int k_offset = b * T * C + j * C;         // K[b,j,:]
        for (int k = 0; k < C; k++) {
            dot += Q[q_offset + k] * K[k_offset + k];
        }
        // Scale by 1/sqrt(C)
        scores[j] = dot / sqrtf((float)C);
    }
    __syncthreads();

    // --- Compute maximum value in scores for numerical stability ---
    float max_val = -1e20f;
    for (int j = tid; j < T; j += blockDim.x) {
        if (scores[j] > max_val)
            max_val = scores[j];
    }
    // Use shared memory for reduction (assume blockDim.x <= THREADS)
    __shared__ float smax[THREADS];
    smax[tid] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (smax[tid + stride] > smax[tid])
                smax[tid] = smax[tid + stride];
        }
        __syncthreads();
    }
    max_val = smax[0];

    // --- Compute exponentials and their sum ---
    float sum = 0.0f;
    for (int j = tid; j < T; j += blockDim.x) {
        float exp_val = expf(scores[j] - max_val);
        scores[j] = exp_val;  // store the exponentiated value back
        sum += exp_val;
    }
    __syncthreads();
    __shared__ float ssum[THREADS];
    ssum[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    sum = ssum[0];

    // --- Normalize scores to form the softmax ---
    for (int j = tid; j < T; j += blockDim.x) {
        scores[j] /= sum;
    }
    __syncthreads();

    // --- Compute weighted sum over V to form the output ---
    // For simplicity, we let thread 0 do the accumulation.
    if (tid == 0) {
        for (int k = 0; k < C; k++) {
            float acc = 0.0f;
            for (int j = 0; j < T; j++) {
                int v_offset = b * T * C + j * C;
                acc += scores[j] * V[v_offset + k];
            }
            out[b * T * C + i * C + k] = acc;
        }
    }
}

// C++ interface for the forward attention function.
torch::Tensor attention_forward(torch::Tensor Q,
                                torch::Tensor K,
                                torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "Only float32 supported");

    int B = Q.size(0);
    int T = Q.size(1);
    int C = Q.size(2);

    auto out = torch::empty({B, T, C}, Q.options());

    // Launch one block per (batch, sequence) pair.
    // Grid dimensions: (B, T), blockDim.x = THREADS.
    // Shared memory size: T * sizeof(float) (for the scores).
    dim3 grid(B, T);
    dim3 block(THREADS);
    size_t shared_memory_size = T * sizeof(float);
    attention_forward_kernel<<<grid, block, shared_memory_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, C
    );
    cudaDeviceSynchronize();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention Forward (CUDA)");
}
