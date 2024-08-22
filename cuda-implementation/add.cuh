#pragma once

void add_forward(
    const void* a, const void* b,
    void* out,
    int N, int block_size,
    bool use_fp16,
    float loss_scale
);

void add_inplace_forward(
    const void* a, void* b,
    int N, int block_size,
    bool use_fp16,
    float loss_scale
);