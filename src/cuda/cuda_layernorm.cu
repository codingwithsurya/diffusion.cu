
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction
__device__ __forceinline__ float warpReduceSum(float val) {
    // Use all 32 threads (full warp)
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using the above warp reduce
__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // one partial sum per warp
    int lane = threadIdx.x % WARP_SIZE; 
    int wid  = threadIdx.x / WARP_SIZE;

    // Each warp does a partial reduction
    val = warpReduceSum(val);

    // Write the reduced value of that warp to shared
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Now, only the first warp needs to reduce across 'wid' entries
    // The first warp is wid == 0
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    // At this point, val is the block-wide sum but
    // only threads in warp0 have the correct value.
    // If you want the final sum to be visible in *every* thread,
    // you need a final broadcast. One approach:

    __shared__ float blockSum;
    if (threadIdx.x == 0) {
        blockSum = val;  // store the final sum in shared[0]
    }
    __syncthreads();

    // Now read it back from shared to "val"
    val = blockSum;
    return val;
}

// -------------------------------------------------------- //
//                   LAYERNORM FORWARD                      //
// -------------------------------------------------------- //
__launch_bounds__(BLOCK_SIZE)
__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    const int N,  // batch size
    const int C   // hidden dimension
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (bid >= N) return;  // guard

    const float* input_row  = input  + bid * C;
    float* output_row       = output + bid * C;

    // ------------------------------------------------
    // Step 1: compute mean for this row
    // ------------------------------------------------
    float sum = 0.0f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        sum += input_row[i];
    }
    sum = blockReduceSum(sum);
    float mu = sum / C;

    // We want all threads in the block to see the same mu
    __shared__ float mu_shared;
    if (tid == 0) {
        mean[bid] = mu;    // optional: store the mean
        mu_shared = mu;    // broadcast via shared memory
    }
    __syncthreads();
    mu = mu_shared;        // now every thread reads mu

    // ------------------------------------------------
    // Step 2: compute variance and rstd
    // ------------------------------------------------
    float var_sum = 0.0f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        float diff = input_row[i] - mu;
        var_sum   += diff * diff;
    }
    var_sum = blockReduceSum(var_sum);
    float sigma     = sqrtf(var_sum / C + 1e-5f);
    float inv_sigma = 1.0f / sigma;

    // Again, broadcast inv_sigma so all threads see it
    __shared__ float inv_sigma_shared;
    if (tid == 0) {
        rstd[bid] = inv_sigma; 
        inv_sigma_shared = inv_sigma;
    }
    __syncthreads();
    inv_sigma = inv_sigma_shared;

    // ------------------------------------------------
    // Step 3: normalize
    // ------------------------------------------------
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        output_row[i] = (input_row[i] - mu) * inv_sigma;
    }
}

torch::Tensor layernorm_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, hidden_size)");
    TORCH_CHECK(input.scalar_type() == torch::ScalarType::Float,
        "Input must be float32");

    const int N = input.size(0);
    const int C = input.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input.device());

    auto output = torch::empty_like(input);
    auto mean   = torch::empty({N}, options);
    auto rstd   = torch::empty({N}, options);

    const dim3 blocks(N);
    const dim3 threads(BLOCK_SIZE);

    layernorm_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        N, C
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
}

