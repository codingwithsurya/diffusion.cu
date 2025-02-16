# test_cuda_mlp.py
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

cuda_mlp = load(
    name="cuda_mlp",
    sources=["/content/cuda_mlp.cu"],
    verbose=True,
    extra_cuda_cflags=['--expt-relaxed-constexpr', '-O3'],
)

def approx_gelu_pytorch(x):
    # replicate the approximate GELU used in the kernel
    return 0.5 * x * (1.0 + torch.tanh(0.79788456*(x + 0.044715* x**3)))

def mlp_ref(x, w1, b1, w2, b2):
    # x: [B, T, C_in]
    # w1: [C_in, C_mid], b1: [C_mid]
    # w2: [C_mid, C_out], b2: [C_out]
    y1 = x.matmul(w1) + b1
    y2 = approx_gelu_pytorch(y1)
    out = y2.matmul(w2) + b2
    return out

def main():
    torch.manual_seed(0)
    B, T, C_in = 2, 5, 16
    C_mid = 32
    C_out = 8
    x = torch.randn(B, T, C_in, device='cuda', dtype=torch.float32)
    w1 = torch.randn(C_in, C_mid, device='cuda', dtype=torch.float32)
    b1 = torch.randn(C_mid, device='cuda', dtype=torch.float32)
    w2 = torch.randn(C_mid, C_out, device='cuda', dtype=torch.float32)
    b2 = torch.randn(C_out, device='cuda', dtype=torch.float32)

    out_cuda = cuda_mlp.mlp_forward(x, w1, b1, w2, b2)
    out_ref = mlp_ref(x, w1, b1, w2, b2)

    max_error = (out_cuda - out_ref).abs().max().item()
    print("Max abs error:", max_error)
    assert torch.allclose(out_cuda, out_ref, atol=1e-3), "Mismatch in MLP"
    print("MLP kernel test passed!")

if __name__ == "__main__":
    main()
