# cuda_dit_block_TEST.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


dit_block_mod = load(
    name="dit_block_mod",
    sources=[
        "cuda_dit_block.cu",
        "cuda_adaln_modulation.cu",
        "attention.cu",
        "mlp.cu",
        "cuda_timestep_embed.cu",
        "cuda_util_kernels.cu",
        "layernorm.cu",
        "label_embed.cu",
        "final_layer.cu"
    ],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

# Pytorch implementation of the DiT block for reference
def modulate(x, shift, scale):
    # x shape: [B, T, C], shift & scale: [B, C]
    # For reference, we'll broadcast over T dimension:
    return (x * scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=1, qkv_bias=True, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x):
        # naive single-head for demonstration
        B, T, C = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C**0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = self.out_proj(out)
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x

###############################################################################
# 3) TEST COMPARISON: CUDA vs. PYTORCH
###############################################################################
def main():
    torch.manual_seed(0)
    B, T, C = 2, 3, 4
    num_heads = 1

    # Random input x, condition c
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    c = torch.randn(B, C, device='cuda', dtype=torch.float32)

    # We'll create random weights for our CUDA call.
    # w_mod: [C, 6*C], b_mod: [6*C]
    w_mod = torch.randn(C, 6*C, device='cuda', dtype=torch.float32)
    b_mod = torch.randn(6*C, device='cuda', dtype=torch.float32)

    # Q,K,V => [C,C], bias => [C], same for out
    wQ = torch.randn(C, C, device='cuda', dtype=torch.float32)
    bQ = torch.randn(C, device='cuda', dtype=torch.float32)
    wK = torch.randn(C, C, device='cuda', dtype=torch.float32)
    bK = torch.randn(C, device='cuda', dtype=torch.float32)
    wV = torch.randn(C, C, device='cuda', dtype=torch.float32)
    bV = torch.randn(C, device='cuda', dtype=torch.float32)
    wO = torch.randn(C, C, device='cuda', dtype=torch.float32)
    bO = torch.randn(C, device='cuda', dtype=torch.float32)

    # MLP => w1: [C, 4C], b1: [4C], w2: [4C, C], b2: [C]
    w1 = torch.randn(C, 4*C, device='cuda', dtype=torch.float32)
    b1 = torch.randn(4*C, device='cuda', dtype=torch.float32)
    w2 = torch.randn(4*C, C, device='cuda', dtype=torch.float32)
    b2 = torch.randn(C, device='cuda', dtype=torch.float32)

    # Run custom CUDA
    out_cuda = dit_block_mod.dit_block_forward(
        x.clone(), c,
        w_mod, b_mod,
        wQ, bQ,
        wK, bK,
        wV, bV,
        wO, bO,
        w1, b1,
        w2, b2,
        1e-6
    )

    # Build the PyTorch reference block
    block_ref = DiTBlock(C, num_heads=num_heads, mlp_ratio=4.0).cuda()

    # Load the weights carefully to match shapes:
    with torch.no_grad():
        # "adaLN_modulation" = nn.Sequential(SiLU, Linear(C->6C))
        # The linear has shape [6C, C] in PyTorch, so we do transpose
        block_ref.adaLN_modulation[1].weight.copy_(w_mod.transpose(0,1))
        block_ref.adaLN_modulation[1].bias.copy_(b_mod)

        # attn.q_proj => [C, C], PyTorch => [C, C] in .weight, but stored [C_out, C_in]
        block_ref.attn.q_proj.weight.copy_(wQ.transpose(0,1))
        block_ref.attn.q_proj.bias.copy_(bQ)
        block_ref.attn.k_proj.weight.copy_(wK.transpose(0,1))
        block_ref.attn.k_proj.bias.copy_(bK)
        block_ref.attn.v_proj.weight.copy_(wV.transpose(0,1))
        block_ref.attn.v_proj.bias.copy_(bV)
        block_ref.attn.out_proj.weight.copy_(wO.transpose(0,1))
        block_ref.attn.out_proj.bias.copy_(bO)

        # MLP
        # fc1 => [4C, C] in PyTorch, so transpose
        block_ref.mlp.fc1.weight.copy_(w1.transpose(0,1))
        block_ref.mlp.fc1.bias.copy_(b1)
        # fc2 => [C, 4C], so again transpose
        block_ref.mlp.fc2.weight.copy_(w2.transpose(0,1))
        block_ref.mlp.fc2.bias.copy_(b2)

    out_ref = block_ref(x.clone(), c)

    # Compare
    max_err = (out_cuda - out_ref).abs().max().item()
    print("DiTBlock test max error:", max_err)
    assert torch.allclose(out_cuda, out_ref, atol=1e-3), "Mismatch in DiT block forward!"
    print("DiT block forward test passed!")

if __name__ == "__main__":
    main()