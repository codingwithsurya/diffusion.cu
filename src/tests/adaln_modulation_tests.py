# test_cuda_adaln_modulation.py
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

cuda_adaln = load(
    name="cuda_adaln",
    sources=["cuda_adaln_modulation.cu"],
    verbose=True,
    extra_cuda_cflags=['--expt-relaxed-constexpr', '-O3']
)

def silu_linear_ref(c, W, b):
    # c: [B, d], W: [d, out_dim], b: [out_dim]
    c_silu = torch.nn.functional.silu(c)
    out = c_silu @ W + b
    return out

def adaln_ref(x, shift, scale, eps=1e-6):
    # x: [B, T, C], shift, scale: [B, C]
    # for each b in [0..B-1], each t in [0..T-1], LN over dimension C
    B, T, C = x.shape
    x_ = x.detach().clone()
    for b_ in range(B):
        for t_ in range(T):
            row = x_[b_, t_]
            mean = row.mean()
            var = row.var(unbiased=False)
            row_ = (row - mean) / torch.sqrt(var + eps)
            row_ = row_ * scale[b_] + shift[b_]
            x_[b_, t_] = row_
    return x_

def main():
    torch.manual_seed(0)
    B, d = 4, 16
    out_dim = 32
    c = torch.randn(B, d, device='cuda', dtype=torch.float32)
    W = torch.randn(d, out_dim, device='cuda', dtype=torch.float32)
    b = torch.randn(out_dim, device='cuda', dtype=torch.float32)

    out_cuda = cuda_adaln.silu_linear_forward(c, W, b)
    out_ref = silu_linear_ref(c, W, b)
    err = (out_cuda - out_ref).abs().max().item()
    print("SiLU + Linear test error:", err)
    assert torch.allclose(out_cuda, out_ref, atol=1e-4)

    # test adaln
    B, T, C = 2, 3, 4
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    shift = torch.randn(B, C, device='cuda', dtype=torch.float32)
    scale = torch.randn(B, C, device='cuda', dtype=torch.float32)
    y_cuda = cuda_adaln.adaln_forward(x, shift, scale, 1e-6)
    y_ref = adaln_ref(x, shift, scale)

    err = (y_cuda - y_ref).abs().max().item()
    print("AdaLN test error:", err)
    assert torch.allclose(y_cuda, y_ref, atol=1e-4)
    print("All adaLN tests passed!")

if __name__ == "__main__":
    main()
