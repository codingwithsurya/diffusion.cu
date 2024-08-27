#include "patch_embed.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "utils.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// ----------------------------------------------------------------------------
// EmbedPatches

// assume img is of shape (B, C, H, W), with H and W divisible by patch_size
// output will be of shape (B, N, C * patch_size^2), where N = (H * W) / (patch_size^2)
// Assume one thread per patch in output
__global__ void embed_patches_forward_kernel(
    const float *img,
    float *out,
    int B, int C, int H, int W, int patch_size)
{
    // get index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_sz = blockDim.x * gridDim.x;

    int img_size = H * W;
    int patch_area = patch_size * patch_size;
    int N = img_size / patch_area;

    for (int i = idx; i < B * N * patch_area * C; i += grid_sz)
    {
        // assume output is of shape (B, N, C, Ph, Pw)
        int b = i / (N * patch_area * C);
        int n = (i / (patch_area * C)) % N;
        int c = (i / patch_area) % C;
        int ph = (i / patch_size) % patch_size;
        int pw = i % patch_size;

        // calculate h and w from n, ph, pw
        int h = (n / (W / patch_size)) * patch_size + ph;
        int w = (n % (W / patch_size)) * patch_size + pw;

        out[i] = img[b * C * img_size + c * img_size + h * W + w];
    }
}

// backwards of embed_patches is similar to backwards of avgpool
// each thread gets one pixel in the original image, and sums
// over the gradients from all patches that use it
__global__ void embed_patches_backward_kernel(
    const float *dout,
    float *dimg,
    int B, int C, int H, int W, int patch_size)
{
    // get index
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_sz = blockDim.x * gridDim.x;

    int img_size = H * W;
    int patch_area = patch_size * patch_size;
    int N = img_size / patch_area;

    for (int i = idx; i < B * img_size * C; i += grid_sz)
    {
        // img is of shape (B, C, H, W)
        int b = i / (C * img_size);
        int c = (i / img_size) % C;
        int h = (i / W) % H;
        int w = i % W;

        // get the patches that use this pixel
        int n_start = (h / patch_size) * (W / patch_size) + w / patch_size;
        int ph_start = h % patch_size;
        int pw_start = w % patch_size;

        float sum = 0.0f;
        for (int n = n_start; n < n_start + 1; n++)
        {
            for (int ph = ph_start; ph < ph_start + 1; ph++)
            {
                for (int pw = pw_start; pw < pw_start + 1; pw++)
                {
                    int dout_idx = b * C * N * patch_area + n * C * patch_area + c * patch_area + ph * patch_size + pw;
                    sum += dout[dout_idx];
                }
            }
        }
        dimg[i] = sum;
    }
}

// ----------------------------------------------------------------------------
// kernel launcher
void embed_patches_forward(
    const float *img,
    float *out,
    int B, int C, int H, int W, int patch_size, int block_size)
{
    assert(H % patch_size == 0);
    assert(W % patch_size == 0);

    int img_size = H * W;
    int patch_area = patch_size * patch_size;
    int N = img_size / patch_area;

    int n_blocks = ceil_div(B * N * patch_area * C, block_size);
    embed_patches_forward_kernel<<<n_blocks, block_size>>>(
        img, out, B, C, H, W, patch_size);
    cudaCheck(cudaGetLastError());
}

void embed_patches_backward(
    const float *dout,
    float *dimg,
    int B, int C, int H, int W, int patch_size, int block_size)
{
    int n_blocks = ceil_div(B * C * H * W, block_size);
    embed_patches_backward_kernel<<<n_blocks, block_size>>>(
        dout, dimg, B, C, H, W, patch_size);
    cudaCheck(cudaGetLastError());
}

#ifndef LINKING
int main(int argc, char **argv)
{
    srand(0);
    int B = 2;
    int C = 3;
    int H = 64;
    int W = 64;
    int patch_size = 8;
    int block_size = 512;

    // get the output shape
    int img_size = H * W;
    int patch_area = patch_size * patch_size;
    int N = img_size / patch_area;

    // create host memory
    float *img = (float *)malloc(B * C * H * W * sizeof(float));
    float *out = (float *)malloc(B * N * patch_area * C * sizeof(float));
    float *dout = (float *)malloc(B * N * patch_area * C * sizeof(float));
    float *dimg = (float *)malloc(B * C * H * W * sizeof(float));
    // read saved output
    FILE *file = fopen("patch_embed.bin", "rb");
    if (!file)
    {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(img, sizeof(float), B * C * H * W, file);
    freadCheck(out, sizeof(float), B * N * patch_area * C, file);
    freadCheck(dout, sizeof(float), B * N * patch_area * C, file);
    freadCheck(dimg, sizeof(float), B * C * H * W, file);
    fclose(file);

    // allocate device memory
    float *d_img, *d_out, *d_dout, *d_dimg;
    cudaCheck(cudaMalloc(&d_img, B * C * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * N * patch_area * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * N * patch_area * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dimg, B * C * H * W * sizeof(float)));
    // copy input to device
    cudaCheck(cudaMemcpy(d_img, img, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * N * patch_area * C * sizeof(float), cudaMemcpyHostToDevice));

    printf("Checking forward pass\n");
    embed_patches_forward(d_img, d_out, B, C, H, W, patch_size, block_size);
    validate_result(d_out, out, "out", B * N * patch_area * C);
    printf("Forward pass successful\n\n");

    printf("Checking backward pass\n");
    embed_patches_backward(d_dout, d_dimg, B, C, H, W, patch_size, block_size);
    validate_result(d_dimg, dimg, "dimg", B * C * H * W);
    printf("Backward pass successful\n\n");
}
#endif