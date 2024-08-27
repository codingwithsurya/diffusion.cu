#include "mlp.cuh"
#include "silu.cuh"
#include "linear.cuh"
#include "utils.h"
#include <assert.h>

void mlp_forward(
    cublasHandle_t cublas_handle,
    float *out, const float *inp,
    float *fc1_w, float *fc2_w,
    int B, int T, int C, int hidden_dim, int block_size)
{
    // project to 4h
    float *tmp;
    cudaCheck(cudaMalloc(&tmp, B * T * hidden_dim * sizeof(float)));
    // Note: there is no bias in DiT MLP
    matmul_forward2(cublas_handle, tmp, inp, fc1_w, nullptr, B * T, C, hidden_dim, block_size);

    // SiLU
    silu_forward(tmp, out, B * T * hidden_dim, block_size);

    // project back to h
    // Note: there is no bias in DiT MLP
    matmul_forward2(cublas_handle, out, out, fc2_w, nullptr, B * T, hidden_dim, C, block_size);

    // add residual, store in out
    add_inplace_forward(inp, out, B * T * C, block_size);
    cudaCheck(cudaFree(tmp));
}

void mlp_backward(
    cublasHandle_t cublas_handle,
    const float *dout, const float *inp,
    float *fc1_w, float *fc2_w,
    float *dfc1_w, float *dfc2_w,
    float *act_buf1, float *act_buf2,
    int B, int T, int C, int hidden_dim, int block_size)
{
    // act_buf2 will be used as a (B, T, hidden_dim) buffer for backward
    float *buf_BT4C = act_buf2;
    float *buf_BTC = act_buf1;

    // dout += inp; // residual connection, taken care of later
    // matmul backward for fc2
    matmul_backward1(cublas_handle, buf_BT4C, dfc2_w, nullptr, dout, act_buf1, fc2_w, B * T, hidden_dim, C);
    // silu backward
    silu_backward(buf_BT4C, act_buf1, buf_BTC, B * T * hidden_dim, block_size);
    // matmul backward for fc1
    matmul_backward1(cublas_handle, buf_BTC, dfc1_w, nullptr, buf_BTC, inp, fc1_w, B * T, C, hidden_dim);

    // Add residual gradient
    add_inplace_forward(dout, buf_BTC, B * T * C, block_size);
}

#ifndef LINKING
// unit test code, will be removed in the actual training script
int main(int argc, char **argv)
{
    setup_main();

    // set up cublas
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));

    int B = 2;
    int T = 1024;
    int C = 768;
    int hidden_dim = C * 4;
    int block_size = 512;

    // allocate memory on host
    float *inp = (float *)malloc(B * T * C * sizeof(float));
    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *dout = (float *)malloc(B * T * C * sizeof(float));
    float *fc1_w = (float *)malloc(hidden_dim * C * sizeof(float));
    float *fc2_w = (float *)malloc(C * hidden_dim * sizeof(float));
    float *dfc1_w = (float *)malloc(hidden_dim * C * sizeof(float));
    float *dfc2_w = (float *)malloc(C * hidden_dim * sizeof(float));
    float *act_buf1 = (float *)malloc(B * T * hidden_dim * sizeof(float));
    float *act_buf2 = (float *)malloc(B * T * hidden_dim * sizeof(float));

    // read saved data from file
    FILE *file = fopen("mlp.bin", "rb");
    if (!file)
    {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(inp, sizeof(float), B * T * C, file);
    freadCheck(out, sizeof(float), B * T * C, file);
    freadCheck(dout, sizeof(float), B * T * C, file);
    freadCheck(fc1_w, sizeof(float), hidden_dim * C, file);
    freadCheck(fc2_w, sizeof(float), C * hidden_dim, file);
    freadCheck(dfc1_w, sizeof(float), hidden_dim * C, file);
    freadCheck(dfc2_w, sizeof(float), C * hidden_dim, file);
    fclose(file);

    // allocate memory on device
    float *d_inp, *d_out, *d_dout, *d_fc1_w, *d_fc2_w, *d_dfc1_w, *d_dfc2_w, *d_act_buf1, *d_act_buf2;
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_fc1_w, hidden_dim * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_fc2_w, C * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dfc1_w, hidden_dim * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dfc2_w, C * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_act_buf1, B * T * hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_act_buf2, B * T * hidden_dim * sizeof(float)));
    // copy from host to device
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_fc1_w, fc1_w, hidden_dim * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_fc2_w, fc2_w, C * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    // run forward and backward
    mlp_forward(cublas_handle, d_out, d_inp, d_fc1_w, d_fc2_w, B, T, C, hidden_dim, block_size);
    mlp_backward(cublas_handle, d_dout, d_inp, d_fc1_w, d_fc2_w, d_dfc1_w, d_dfc2_w, d_act_buf1, d_act_buf2, B, T, C, hidden_dim, block_size);

    // copy results back to host
    cudaCheck(cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dfc1_w, d_dfc1_w, hidden_dim * C * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(dfc2_w, d_dfc2_w, C * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    // verify results
    float acc = 1e-3;
    printf("Checking forward pass\n");
    validate_result(out, out, "out", B * T * C, acc);
    printf("\nChecking backward pass\n");
    printf("Checking dfc1_w\n");
    validate_result(dfc1_w, dfc1_w, "dfc1_w", hidden_dim * C, acc);
    printf("Checking dfc2_w\n");
    validate_result(dfc2_w, dfc2_w, "dfc2_w", C * hidden_dim, acc);

    // free memory on host
    free(inp);
    free(out);
    free(dout);
    free(fc1_w);
    free(fc2_w);
    free(dfc1_w);
    free(dfc2_w);
    free(act_buf1);
    free(act_buf2);

    // free memory on device
    cudaFree(d_inp);
    cudaFree(d_out);
    cudaFree(d_dout);
    cudaFree(d_fc1_w);
    cudaFree(d_fc2_w);
    cudaFree(d_dfc1_w);
    cudaFree(d_dfc2_w);
    cudaFree(d_act_buf1);
    cudaFree(d_act_buf2);

    // free cublas handle
    cublasDestroy(cublas_handle);
}
#endif