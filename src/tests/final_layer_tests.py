# test_final_layer.py
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Adjust the file path if needed
final_layer_mod = load(
    name="final_layer_mod",
    sources=["final_layer.cu"],
    extra_cuda_cflags=['-O3'],
    verbose=True
)

def final_layer_ref(x, c, w_mod, b_mod, w_linear, b_linear, eps=1e-6):
    """
    Pure PyTorch reference for the final layer:
    1) shift, scale = chunk(2, dim=1) of [B, 2*C]
    2) LN x => (x - mean)/sqrt(var+eps), no gamma/beta
    3) out = (x_ln * scale + shift) @ w_linear + b_linear
    """
    B, T, C = x.shape
    out_dim = w_linear.shape[1]

    # 1) mod_out = Linear(SiLU(c)) => [B, 2*C]
    c_silu = F.silu(c)
    mod_out = c_silu @ w_mod + b_mod
    shift, scale = mod_out.chunk(2, dim=1)  # each [B, C]

    # 2) LN
    # shape [B, T, C], per-(b,t)
    mu = x.mean(dim=2, keepdim=True)
    var = x.var(dim=2, keepdim=True, unbiased=False)
    x_ln = (x - mu) / torch.sqrt(var + eps)

    # 3) out = linear( (x_ln * scale + shift) )
    # broadcast scale, shift over T
    x_mod = x_ln * scale.unsqueeze(1) + shift.unsqueeze(1)
    out = x_mod.reshape(B*T, C) @ w_linear + b_linear
    return out.reshape(B, T, out_dim)

def main():
    torch.manual_seed(0)
    B, T, C = 2, 3, 4
    out_dim = 5
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    c = torch.randn(B, C, device='cuda', dtype=torch.float32)

    w_mod = torch.randn(C, 2*C, device='cuda', dtype=torch.float32)
    b_mod = torch.randn(2*C, device='cuda', dtype=torch.float32)
    w_linear = torch.randn(C, out_dim, device='cuda', dtype=torch.float32)
    b_linear = torch.randn(out_dim, device='cuda', dtype=torch.float32)

    out_cuda = final_layer_mod.final_layer_forward(
        x, c, w_mod, b_mod, w_linear, b_linear, 1e-6
    )
    out_ref = final_layer_ref(x, c, w_mod, b_mod, w_linear, b_linear, 1e-6)
    max_err = (out_cuda - out_ref).abs().max().item()
    print("Final Layer - max error:", max_err)
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), "Mismatch!"
    print("Final layer test passed!")

if __name__ == "__main__":
    main()