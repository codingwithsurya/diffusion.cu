#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <type_traits>

#include "common.h"
#include "layernorm.cuh"

// ----------------------------------------------------------------------------
// Adapted from llm.c layernorm implementation
// https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu

// uses shared memory instead for the reduces
// template to allow for different data types for input, output, weights, biases
// this is needed because we want to be able to use mixed precision, e.g.,
// bf16 for inputs and outputs, but fp32 for weights and biases.
// most similar to llm.c's kernel 10
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
__global__ void layernorm_forward_kernel1(
    Tdinp *__restrict__ out, Trest *__restrict__ mean, Trest *__restrict__ rstd,
    const Tdinp *__restrict__ inp, const Tparams *__restrict__ weight,
    const Tparams *__restrict__ bias, const float *__restrict__ c, int N, int C)
{
    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; // number of warps in block
    extern __shared__ float shared[];

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = CEIL_DIV(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float *s_mean = shared;
    float *s_var = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float *s_mean_tmp = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float *s_var_tmp = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for (int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size)
    {
        store128(s_mean + i, f128::zeros());
        store128(s_var + i, f128::zeros());
    }
    __syncthreads();

    for (int idx = baseIdx; idx < N; idx += warpsInGrid)
    {
        const Tdinp *inp_idx = inp + idx * C;
        Tdinp *out_idx = out + idx * C;

        // first: two reduce operations
        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size)
        {
            x128 packed_inp = load128(inp_idx + i);
            for (int k = 0; k < x128::size; k++)
            {
                sum += (float)packed_inp[k];
                sum2 += (float)packed_inp[k] * (float)packed_inp[k];
            }
        }
        sum = warpReduceSum(sum) / C;
        sum2 = warpReduceSum(sum2) / C;

        // mean, var, rstd
        float m = sum;
        float var = sum2 - sum * sum;
        float s = rsqrtf(var + 1e-5f);

        if (warpThreadIdx == 0 && mean != nullptr)
        {
            __stcs(mean + idx, m);
        }
        if (warpThreadIdx == 0 && rstd != nullptr)
        {
            __stcs(rstd + idx, s);
        }

        // compute shift and scale from conditioning vector `c`
        float shift = c[idx * 2 + 0];
        float scale = c[idx * 2 + 1];

        // final normalization and scaling by weight/bias
        for (int c = 0; c < iterations_C; c++)
        {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C)
            {
                break;
            }

            x128 w128 = load128(weight + global_index);
            x128 b128 = load128(bias + global_index);
            x128 in128 = load128(inp_idx + global_index);
            x128 out128;
            for (int k = 0; k < x128::size; ++k)
            {
                // apply shift and scale from `c` before weight and bias
                out128[k] = (Tdinp)((((float)in128[k] - m) * s * scale + shift) * (float)w128[k] + (float)b128[k]);
            }

            store128(out_idx + global_index, out128);
        }
    }
}

// no template for backward, because we will always use FP32 for dbias, dweight
// most similar to kernel 10 of layernorm_backward in llm.c
template <typename Tdinp, typename Tdout, typename Trest>
__global__ void __launch_bounds__(512, 2) // todo - any warnings on Turing with only 1024 threads?
    layernorm_backward_kernel1(float *dinp, float *dweight, float *dbias, float *scratch,
                               const Tdout *dout, const Trest *inp, const float *weight,
                               const Trest *mean, const Trest *rstd, const float *c,
                               int B, int T, int C)
{
    int BLOCK_SIZE = blockDim.x;
    int warpsInBlock = BLOCK_SIZE / WARP_SIZE; // number of warps in block
    extern __shared__ float shared[];

    int warpId = threadIdx.x / WARP_SIZE; // warp index within a block
    int baseIdx = blockIdx.x * warpsInBlock + warpId;
    int warpThreadIdx = threadIdx.x % WARP_SIZE; // Thread index within the warp
    int warpsInGrid = gridDim.x * warpsInBlock;
    int C_per_iteration = WARP_SIZE * x128::size;
    int iterations_C = CEIL_DIV(C, C_per_iteration); // + 2;

    // the first half of shared memory is bias, second is weight
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    float *dbias_shared = shared;
    float *dweight_shared = shared + rounded_C;
    // warp zero doesn't actually write to the _tmp_shared memory locations, so we don't need to reserve memory
    // the obvious solution is to change the addressing below to use (threadId.x-32) as offset, but that causes
    // register spills, so instead we mess with the base pointer here, which doesn't increase register usage.
    float *dbias_tmp_shared = shared + 2 * rounded_C - WARP_SIZE * f128::size;
    float *dweight_tmp_shared = shared + 2 * rounded_C + f128::size * BLOCK_SIZE - 2 * WARP_SIZE * f128::size;

    // init shared memory to zero
    for (int i = threadIdx.x * f128::size; i < rounded_C; i += BLOCK_SIZE * f128::size)
    {
        store128(dbias_shared + i, f128::zeros());
        store128(dweight_shared + i, f128::zeros());
    }
    __syncthreads();

    for (int bt = baseIdx; bt < B * T; bt += warpsInGrid)
    {
        const Tdout *dout_bt = dout + bt * C;
        const Trest *inp_bt = inp + bt * C;
        float *dinp_bt = dinp + bt * C;

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = warpThreadIdx * x128::size; i < C; i += WARP_SIZE * x128::size)
        {
            x128 dout128_i = load128(dout_bt + i);
            x128 inp128_i = load128(inp_bt + i);
            x128 weight128_i = load128(weight + i);
            for (int k = 0; k < x128::size; k++)
            {
                float dnorm_i = (float)weight128_i[k] * (float)dout128_i[k];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * (float)inp128_i[k];
            }
        }

        const float mean_bt = (float)mean[bt];
        const float rstd_bt = (float)rstd[bt];
        dnorm_mean = warpReduceSum(dnorm_mean) / C;
        dnorm_norm_mean = warpReduceSum(dnorm_norm_mean) / C * rstd_bt - dnorm_mean * mean_bt * rstd_bt;

        // extract shift and scale from the conditioning vector `c`
        float shift = c[bt * 2 + 0];
        float scale = c[bt * 2 + 1];

        for (int c = 0; c < iterations_C; c++)
        {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);

            x128 dout128 = x128::zeros();
            x128 inp128 = x128::zeros();
            x128 weight128 = x128::zeros();

            if (global_index < C)
            {
                dout128 = load128cs(dout_bt + global_index);
                inp128 = load128cs(inp_bt + global_index);
                weight128 = load128(weight + global_index);
            }

            for (int o = 0; o < x128::size / f128::size; ++o)
            {
                f128 dbias_f;
                f128 dweight_f;
                for (int i = 0; i < f128::size; ++i)
                {
                    int x = o * f128::size + i;
                    float dout_i = (float)dout128[x];
                    float norm_bti = ((float)inp128[x] - mean_bt) * rstd_bt;
                    dbias_f[i] = dout_i * weight[global_index + x] * scale; // include weight and scale from `c` here
                    dweight_f[i] = norm_bti * dout_i * scale;               // include scale from `c` here
                }

                if (warpId != 0)
                {
                    store128(dbias_tmp_shared + threadIdx.x * f128::size, dbias_f);
                    // this seems to generate a 64-bit store, instead of 128-bit.
                    // however, forcing 128-bit (e.g., using inline ptx), results in register
                    // spilling and much worse performance, so we'll keep it like this for now
                    // but ideally, we could reduce the register pressure a little.
                    store128(dweight_tmp_shared + threadIdx.x * f128::size, dweight_f);
                }
                __syncthreads();
                if (warpId == 0)
                {
                    for (int j = 1; j < warpsInBlock; j++)
                    {
                        f128 dbias_tmp = load128(dbias_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        f128 dweight_tmp = load128(dweight_tmp_shared + f128::size * (threadIdx.x + j * WARP_SIZE));
                        for (int i = 0; i < f128::size; ++i)
                        {
                            dbias_f[i] += dbias_tmp[i];
                            dweight_f[i] += dweight_tmp[i];
                        }
                    }
                }
                __syncthreads();
                if (warpId == 0)
                {
                    f128 db_old = load128(dbias_shared + global_index + f128::size * o);
                    f128 dw_old = load128(dweight_shared + global_index + f128::size * o);
                    for (int i = 0; i < f128::size; ++i)
                    {
                        dbias_f[i] += db_old[i];
                        dweight_f[i] += dw_old[i];
                    }
                    store128(dbias_shared + global_index + f128::size * o, dbias_f);
                    store128(dweight_shared + global_index + f128::size * o, dweight_f);
                }
            }
        }
    }
    __syncthreads();
    // Each block writes its partial sum to global memory
    // The last block to finish becomes responsible for summing up all the partial sums
    // This is done by atomically incrementing a flag (cleared to 0 before launching the kernel)
    unsigned int *scratchFlag = (unsigned int *)(scratch);
    // Increment scratch pointer by a full cacheline so that everything remains cacheline aligned
    scratch += 32;
    float *scratch_dbias = scratch;
    float *scratch_dweight = scratch + C;
    for (int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size)
    {
        // Write to global memory in the same "shared memory banking friendly" order
        store128(scratch_dbias + i + 2 * C * blockIdx.x, load128(dbias_shared + i));
        store128(scratch_dweight + i + 2 * C * blockIdx.x, load128(dweight_shared + i));
    }
    __syncthreads();
    // that portion of shared memory is no longer used, so we can repurpose it for the scratch flag.
    unsigned int *tmp_flag = (unsigned int *)(shared + 2 * rounded_C);
    if (threadIdx.x == 0)
    {
        *tmp_flag = atomicInc(scratchFlag, gridDim.x);
    }
    __syncthreads();
    if (*tmp_flag == gridDim.x - 1)
    {
        // Reduction of the partial sums by the final block
        // todo - there isn't enough parallelism even inside that single SM...
        // ==> so could maybe split into another kernel with YET ANOTHER level of reduction?!
        for (int i = threadIdx.x * f128::size; i < C; i += BLOCK_SIZE * f128::size)
        {
            f128 dbias_accum = f128::zeros();
            f128 dweight_accum = f128::zeros();

            for (int read_block_idx = 0; read_block_idx < gridDim.x; read_block_idx++)
            {
                int offset = i + 2 * C * read_block_idx;
                f128 dbias128 = load128(scratch_dbias + offset);
                f128 dweight128 = load128(scratch_dweight + offset);
                for (int k = 0; k < f128::size; k++)
                {
                    dbias_accum[k] += dbias128[k];
                    dweight_accum[k] += dweight128[k];
                }
            }
            store128(dbias_shared + i, dbias_accum);
            store128(dweight_shared + i, dweight_accum);
        }
        __syncthreads();

        // convert from float/FP32 to floatX/BF16 for the final write
        // this is separate because it cannot use as many warps as the above (f128 vs x128)
        // todo - if we split this code into another kernel, we could maybe do it at the same time?
        for (int c = warpId; c < iterations_C; c += warpsInBlock)
        {
            int global_index = (warpThreadIdx * x128::size) + (c * C_per_iteration);
            if (global_index >= C)
            {
                break;
            }

            x128 dbias128 = load128(dbias + global_index);
            x128 dweight128 = load128(dweight + global_index);
            for (int o = 0; o < x128::size / f128::size; ++o)
            {
                f128 s_db = load128(dbias_shared + global_index + o * f128::size);
                f128 s_dw = load128(dweight_shared + global_index + o * f128::size);
                for (int i = 0; i < f128::size; ++i)
                {
                    int x = o * f128::size + i;
                    dbias128[x] = (floatX)(s_db[i] + (float)dbias128[x]);
                    dweight128[x] = (floatX)(s_dw[i] + (float)dweight128[x]);
                }
            }
            store128(dbias + global_index, dbias128);
            store128(dweight + global_index, dweight128);
        }
    }

    // update dinp
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE)
    {
        float norm_bti = ((float)inp_bt[i] - mean_bt) * rstd_bt;
        float dval = 0.0f;
        dval += (float)weight[i] * (float)dout_bt[i]; // term 1
        dval -= dnorm_mean;                           // term 2
        dval -= norm_bti * dnorm_norm_mean;           // term 3
        dval *= rstd_bt * scale;                      // include scale from `c` in the final scaling factor
        dinp_bt[i] = (Tdinp)((float)dinp_bt[i] + dval);
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_forward1(Tdinp *out, Trest *mean, Trest *rstd,
                        const Tdinp *inp, const Tparams *weight, const Tparams *bias, const float *c,
                        int N, int C,
                        const int block_size, cudaStream_t stream)
{
    const int grid_size = ceil_div(N * 32, block_size);
    // allocate shared memory for kernel 1
    size_t smem = (2 * CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size) + 2 * (block_size - 32) * f128::size) * sizeof(float);
    layernorm_forward_kernel1<<<grid_size, block_size, smem, stream>>>(out, mean, rstd, inp, weight, bias, c, N, C);
    cudaCheck(cudaGetLastError());
}

template <typename Tdinp, typename Tdout, typename Trest>
void layernorm_backward1(Tdinp *dinp, float *dweight, float *dbias, float *scratch,
                         const Tdout *dout, const Trest *inp, const float *weight, const Trest *mean, const Trest *rstd, const float *c,
                         int B, int T, int C, int block_size, cudaStream_t stream)
{
    const int grid_size = (1024 / block_size) * cuda_num_SMs; // todo - heuristics for other GPUs?
    size_t rounded_C = CEIL_DIV(C, (32 * x128::size)) * (32 * x128::size);
    size_t shared_mem_size = (2 * rounded_C + 2 * (block_size - 32) * f128::size) * sizeof(float);

    cudaCheck(cudaMemset(scratch, 0, 1 * sizeof(float))); // just need to memset the flag for this version
    layernorm_backward_kernel1<<<grid_size, block_size, shared_mem_size, stream>>>(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, c, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_forward(Tdinp *out, Trest *mean, Trest *rstd,
                       const Tdinp *inp, const Tparams *weight, const Tparams *bias, const float *c,
                       int N, int C,
                       const int block_size, cudaStream_t stream)
{
    layernorm_forward1(out, mean, rstd, inp, weight, bias, c, N, C, block_size, stream);
}

template <typename Tdinp, typename Tdout, typename Trest>
void layernorm_backward(Tdinp *dinp, float *dweight, float *dbias, float *scratch,
                        const Tdout *dout, const Trest *inp, const float *weight, const Trest *mean, const Trest *rstd, const float *c,
                        int B, int T, int C, int block_size, cudaStream_t stream)
{
    layernorm_backward1(dinp, dweight, dbias, scratch, dout, inp, weight, mean, rstd, c, B, T, C, block_size, stream);
}

#ifndef LINKING
int main(int argc, char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 1600; // this is the problematic size

    // first do the forward pass in CPU
    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *mean = (float *)malloc(B * T * sizeof(float));
    float *rstd = (float *)malloc(B * T * sizeof(float));
    float *inp = make_random_float(B * T * C);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);
    // allocate memory for conditioning vector `c`
    float *c = (float *)malloc(B * T * 2 * sizeof(float));
    for (int i = 0; i < B * T * 2; i++)
    {
        c[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    layernorm_forward_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    float *dbias = make_zeros_float(C);
    layernorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    floatX *d_dinp;
    floatX *d_dweight;
    floatX *d_dbias;
    floatX *d_dout;
    floatX *d_inp;
    float *d_weight;
    floatX *d_mean;
    floatX *d_rstd;
    float *d_scratch;
    float *d_c;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dbias, C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_scratch, (1024 / 32) * cuda_num_SMs * (2 * C + 32) * sizeof(float)));
    cudaCheck(cudaMalloc(&d_c, B * T * 2 * sizeof(float))); // copy conditioning vector to GPU
    // copy over the "inputs" to the backward call
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(memcpy_convert(d_mean, mean, B * T));
    cudaCheck(cudaMemcpy(d_c, c, B * T * 2 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(memcpy_convert(d_rstd, rstd, B * T));
    // set up cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // launch the kernel
    int block_sizes[] = {32, 64, 128, 256, 512, 768, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(floatX)));
        cudaCheck(cudaMemset(d_dbias, 0, C * sizeof(floatX)));

        layernorm_forward<floatX, float, floatX, float>(d_inp, d_mean, d_rstd, d_inp, d_weight, d_dbias, d_c,
                                                        B * T, C, block_size, stream);

        layernorm_backward<floatX, floatX, float>(d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd, d_c,
                                                  B, T, C, block_size, stream);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(floatX) == 4 ? 1e-3f : 1e-1f;    // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(floatX) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        printf("dbias:\n");
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_backward<floatX, floatX, float>,
                                              d_dinp, d_dweight, d_dbias, d_scratch, d_dout, d_inp, d_weight, d_mean, d_rstd, d_c,
                                              B, T, C, block_size);
        printf("block_size %4d time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanups
    cudaStreamDestroy(stream);
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    free(c);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_mean));
    cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_scratch));
    cudaCheck(cudaFree(d_c));
    return 0;
}
#endif