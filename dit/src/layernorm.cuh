#pragma once
#include <cuda_runtime.h>

// generic function to apply layernorm, with or without weight/bias parameters
// includes the conditioning vector 'c' for adaLN-Zero
template <typename Tdinp, typename Tparams, typename Tdout, typename Trest>
void layernorm_forward(Tdinp *out, Trest *mean, Trest *rstd,
                       const Tdinp *inp, const Tparams *weight, const Tparams *bias, const float *c,
                       int N, int C,
                       const int block_size, cudaStream_t stream);

// backward function for layernorm, using fp32 for gradients
// includes the conditioning vector 'c' for adaLN-Zero
template <typename Tdinp, typename Tdout, typename Trest>
void layernorm_backward(Tdinp *dinp, float *dweight, float *dbias, float *scratch,
                        const Tdout *dout, const Trest *inp, const float *weight, const Trest *mean, const Trest *rstd, const float *c,
                        int B, int T, int C, int block_size, cudaStream_t stream);