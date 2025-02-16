import torch
from torch.utils.cpp_extension import load
import os


# Compile the extension with optimizations for A100
cuda_layernorm = load(
    name="cuda_layernorm",
    sources=["cuda_layernorm.cu"],
    verbose=True,
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-lineinfo'
    ]
)

# Test function
def test_layernorm():
    B, C = 128, 768
    x = torch.randn(B, C, device='cuda', dtype=torch.float32)
    
    # Run our custom implementation
    out_cuda = cuda_layernorm.layernorm_forward(x)
    
    # Run PyTorch's implementation as reference
    layer_norm = torch.nn.LayerNorm(C, elementwise_affine=False).cuda()
    out_ref = layer_norm(x)
    
    # Compare results
    max_error = (out_cuda - out_ref).abs().max().item()
    print(f"Max absolute error: {max_error}")
    assert torch.allclose(out_cuda, out_ref, atol=1e-5)
    print("LayerNorm test passed!")

test_layernorm()