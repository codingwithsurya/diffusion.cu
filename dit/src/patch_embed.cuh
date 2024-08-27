#pragma once

void embed_patches_forward(
    const float *img,
    float *out,
    int B, int C, int H, int W, int patch_size, int block_size);

void embed_patches_backward(
    const float *dout,
    float *dimg,
    int B, int C, int H, int W, int patch_size, int block_size);