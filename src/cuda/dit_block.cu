// cuda_dit_block.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Forward functions
torch::Tensor silu_linear_forward(torch::Tensor c,
                                  torch::Tensor w,
                                  torch::Tensor b);

torch::Tensor adaln_forward(torch::Tensor x,
                            torch::Tensor shift,
                            torch::Tensor scale,
                            float eps);

torch::Tensor attention_forward(torch::Tensor Q,
                                torch::Tensor K,
                                torch::Tensor V);

torch::Tensor mlp_forward(torch::Tensor x,
                          torch::Tensor w1,
                          torch::Tensor b1,
                          torch::Tensor w2,
                          torch::Tensor b2);

torch::Tensor linear_forward(torch::Tensor A,
                             torch::Tensor weight,
                             torch::Tensor bias);

// -----------------------------------------------------------------------------
// dit_block_forward(...)
//
// x: [B, T, C]
// c: [B, C]
// w_mod,b_mod => shape [C, 6*C],[6*C] for the "adaLN_modulation" MLP
// wQ,bQ => [C,C],[C], wK,bK => [C,C],[C], wV,bV => [C,C],[C], wO,bO => [C,C],[C] for attention
// w1,b1 => [C,4C],[4C], w2,b2 => [4C,C],[C] for MLP
//
// Steps:
//   1) c_silu -> linear => chunk(6) => shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
//   2) LN(x, shift_msa, scale_msa) => Q,K,V => attention => out_attn => linear => x += gate_msa * out_attn
//   3) LN(x, shift_mlp, scale_mlp) => MLP => x += gate_mlp * x_mlp
//
// Returns x (updated).
// -----------------------------------------------------------------------------
torch::Tensor dit_block_forward(
    torch::Tensor x,        // [B, T, C]
    torch::Tensor c,        // [B, C]
    torch::Tensor w_mod,    // [C, 6*C]
    torch::Tensor b_mod,    // [6*C]
    // Q,K,V linear weights & biases
    torch::Tensor wQ, torch::Tensor bQ,
    torch::Tensor wK, torch::Tensor bK,
    torch::Tensor wV, torch::Tensor bV,
    // out linear for attention
    torch::Tensor wO, torch::Tensor bO,
    // MLP weights
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    float eps
) {
    // 1) c_silu => linear => shape [B, 6*C]
    auto sixC = silu_linear_forward(c, w_mod, b_mod);
    auto parts = sixC.chunk(6, 1);  // [B, C] each
    auto shift_msa = parts[0];
    auto scale_msa = parts[1];
    auto gate_msa  = parts[2];
    auto shift_mlp = parts[3];
    auto scale_mlp = parts[4];
    auto gate_mlp  = parts[5];

    // 2) LN => Q,K,V => attention => out_attn => gating
    auto x_ln = adaln_forward(x, shift_msa, scale_msa, eps); // [B, T, C]
    int B = x.size(0);
    int T = x.size(1);
    int C_ = x.size(2);

    // Flatten x_ln for the "linear_forward" calls
    auto x_ln_flat = x_ln.reshape({B * T, C_});

    auto Q_ = linear_forward(x_ln_flat, wQ, bQ).reshape({B, T, C_});
    auto K_ = linear_forward(x_ln_flat, wK, bK).reshape({B, T, C_});
    auto V_ = linear_forward(x_ln_flat, wV, bV).reshape({B, T, C_});

    auto attn = attention_forward(Q_, K_, V_); // [B, T, C]
    auto attn_flat = attn.reshape({B * T, C_});
    auto out_attn = linear_forward(attn_flat, wO, bO).reshape({B, T, C_});

    // x = x + gate_msa.unsqueeze(1) * out_attn
    x = x + gate_msa.unsqueeze(1) * out_attn;

    // 3) LN => MLP => gating
    auto x_ln2 = adaln_forward(x, shift_mlp, scale_mlp, eps);
    auto x_mlp = mlp_forward(x_ln2, w1, b1, w2, b2);

    // x = x + gate_mlp.unsqueeze(1) * x_mlp
    x = x + gate_mlp.unsqueeze(1) * x_mlp;

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dit_block_forward", &dit_block_forward, "DiT Block forward (CUDA)");
}