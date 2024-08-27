#include "final_layer.cuh"
#include "utils.cuh"
#include "layernorm.cuh"
#include "linear.cuh"

// ----------------------------------------------------------------------------
// Final Layer

// x: (B, N, C), c: (B, C)
// out: (B, N, patch_size^2 * out_channels)
void final_layer_forward(
    FinalLayer *final_layer,
    const float *x,
    const float *c,
    cublasHandle_t cublas_handle)
{
    int B = final_layer->B;
    int N = final_layer->N;
    int C = final_layer->C;
    int hidden_size = final_layer->hidden_size;

    // Apply adaLN modulation
    // First, project the conditioning vector `c` to get shift and scale
    matmul_forward2(
        cublas_handle,
        final_layer->shift_scale,
        c,
        final_layer->fc.w,
        final_layer->fc.b,
        B,
        C,
        2 * C,
        final_layer->block_size);
    float *shift = final_layer->shift_scale;
    float *scale = final_layer->shift_scale + B * C;

    // Apply layernorm with shift and scale from adaLN
    layernorm_forward<float, float, float, float>(
        final_layer->ln_out,
        final_layer->ln_mean,
        final_layer->ln_rstd,
        x,
        final_layer->ln_w,
        final_layer->ln_b,
        shift,
        scale,
        B * N,
        C,
        final_layer->block_size,
        0);

    // Project to the output dimension
    matmul_forward2(
        cublas_handle,
        final_layer->out,
        final_layer->ln_out,
        final_layer->proj.w,
        final_layer->proj.b,
        B * N,
        C,
        final_layer->out_channels * final_layer->patch_size * final_layer->patch_size,
        final_layer->block_size);
}

void final_layer_backward(
    FinalLayer *final_layer,
    const float *dout,
    const float *x,
    const float *c,
    cublasHandle_t cublas_handle)
{
    int B = final_layer->B;
    int N = final_layer->N;
    int C = final_layer->C;
    int hidden_size = final_layer->hidden_size;

    // Backward through the projection layer
    matmul_backward1(
        cublas_handle,
        final_layer->ln_out,
        final_layer->proj.w,
        final_layer->proj.b,
        dout,
        final_layer->proj.inp,
        final_layer->proj.w,
        B * N,
        C,
        final_layer->out_channels * final_layer->patch_size * final_layer->patch_size);

    // Backward through LayerNorm (including adaLN modulation)
    float *shift = final_layer->shift_scale;
    float *scale = final_layer->shift_scale + B * C;
    layernorm_backward<float, float, float>(
        final_layer->dx,
        final_layer->ln_w,
        final_layer->ln_b,
        final_layer->scratch,
        final_layer->ln_out,
        x,
        final_layer->ln_w,
        final_layer->ln_mean,
        final_layer->ln_rstd,
        shift,
        scale,
        B * N,
        C,
        final_layer->block_size,
        0);

    // Backward through the projection of the conditioning vector `c`
    matmul_backward1(
        cublas_handle,
        final_layer->dc,
        final_layer->fc.w,
        final_layer->fc.b,
        final_layer->shift_scale,
        c,
        final_layer->fc.w,
        B,
        C,
        2 * C);
}

// ----------------------------------------------------------------------------
// memory management

void final_layer_init(
    FinalLayer *final_layer,
    int patch_size,
    int in_channels,
    int out_channels,
    int B,
    int H,
    int W,
    int hidden_size,
    int block_size)
{
    final_layer->patch_size = patch_size;
    final_layer->in_channels = in_channels;
    final_layer->out_channels = out_channels;
    final_layer->B = B;
    final_layer->H = H;
    final_layer->W = W;
    final_layer->hidden_size = hidden_size;
    final_layer->block_size = block_size;
    final_layer->N = H * W / (patch_size * patch_size);
    final_layer->C = hidden_size;

    // Allocate memory for weights and biases
    cudaCheck(cudaMalloc(&(final_layer->fc.w), hidden_size * 2 * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->fc.b), 2 * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->proj.w), hidden_size * out_channels * patch_size * patch_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->proj.b), out_channels * patch_size * patch_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->ln_w), hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->ln_b), hidden_size * sizeof(float)));

    // Allocate memory for intermediate activations
    cudaCheck(cudaMalloc(&(final_layer->shift_scale), B * 2 * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->ln_out), B * final_layer->N * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->ln_mean), B * final_layer->N * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->ln_rstd), B * final_layer->N * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->scratch), (1024 / 32) * cuda_num_SMs * (2 * hidden_size + 32) * sizeof(float)));

    // Allocate memory for output and gradients
    cudaCheck(cudaMalloc(&(final_layer->out), B * final_layer->N * out_channels * patch_size * patch_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->dx), B * final_layer->N * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(final_layer->dc), B * hidden_size * sizeof(float)));

    // Set input pointers for backward pass
    final_layer->fc.inp = c;
    final_layer->proj.inp = final_layer->ln_out;
}

void final_layer_free(FinalLayer *final_layer)
{
    cudaCheck(cudaFree(final_layer->fc.w));
    cudaCheck(cudaFree(final_layer->fc.b));
    cudaCheck(cudaFree(final_layer->proj.w));
    cudaCheck(cudaFree(final_layer->proj.b));
    cudaCheck(cudaFree(final_layer->ln_w));
    cudaCheck(cudaFree(final_layer->ln_b));

    cudaCheck(cudaFree(final_layer->shift_scale));
    cudaCheck(cudaFree(final_layer->ln_out));
    cudaCheck(cudaFree(final_layer->ln_mean));
    cudaCheck(cudaFree(final_layer->ln_rstd));
    cudaCheck(cudaFree(final_layer->scratch));

    cudaCheck(cudaFree(final_layer->out));
    cudaCheck(cudaFree(final_layer->dx));
    cudaCheck(cudaFree(final_layer->dc));
}