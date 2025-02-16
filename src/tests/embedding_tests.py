import torch
from torch.utils.cpp_extension import load
import torch.nn as nn 
import numpy as np
import math 

timestep_embed = load(
    name="cuda_timestep_embed",
    sources=["/content/cuda_timestep_embed.cu"],
    verbose=True,
    extra_cuda_cflags=['--expt-relaxed-constexpr']
)

label_embed = load(
    name="cuda_label_embed",
    sources=["/content/cuda_label_embed.cu"],
    verbose=True,
    extra_cuda_cflags=['--expt-relaxed-constexpr']
)

# --- TimestepEmbedder Reference (copied from models.py) ---
class TimestepEmbedderRef(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# --- Test for TimestepEmbedder ---
N = 32
frequency_embedding_size = 256
hidden_size = 1152
t = torch.rand(N, device='cuda') * 1000  # random timesteps

# Precompute frequencies and move to CUDA.
half = frequency_embedding_size // 2
freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to('cuda')

# Test sinusoidal embedding kernel.
embed_cuda = timestep_embed.timestep_embedding_forward(t, freqs, frequency_embedding_size)
embed_ref = TimestepEmbedderRef.timestep_embedding(t, frequency_embedding_size)
assert torch.allclose(embed_cuda, embed_ref, atol=1e-5), "Timestep embedding kernel failed"
print("Timestep embedding kernel test passed.")

# Prepare random weights for the MLP layers.
weight1 = torch.randn(hidden_size, frequency_embedding_size, device='cuda')
bias1 = torch.randn(hidden_size, device='cuda')
weight2 = torch.randn(hidden_size, hidden_size, device='cuda')
bias2 = torch.randn(hidden_size, device='cuda')

# Create the reference MLP and move it to CUDA.
ref_mlp = TimestepEmbedderRef(hidden_size, frequency_embedding_size).to('cuda')
with torch.no_grad():
    ref_mlp.mlp[0].weight.copy_(weight1)
    ref_mlp.mlp[0].bias.copy_(bias1)
    ref_mlp.mlp[2].weight.copy_(weight2)
    ref_mlp.mlp[2].bias.copy_(bias2)

out_ref = ref_mlp.forward(t)

# Compute MLP output via the CUDA kernels.
t_freq_cuda = embed_cuda  # our computed sinusoidal embedding
hidden_cuda = timestep_embed.linear_silu_forward(t_freq_cuda, weight1, bias1)
out_cuda = timestep_embed.linear_forward(hidden_cuda, weight2, bias2)

assert torch.allclose(out_cuda, out_ref, atol=1e-3), "Timestep embed MLP kernel failed"
print("Timestep embed MLP kernel test passed.")

# --- Test for LabelEmbedder ---
class LabelEmbedderRef(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=True, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

num_classes = 1000
hidden_size = 1152
dropout_prob = 0.1  # disable dropout for testing

N = 32
labels = torch.randint(0, num_classes, (N,), device='cuda')
force_drop = torch.zeros(N, dtype=torch.int32, device='cuda')
embedding_table = torch.randn(num_classes+1, hidden_size, device='cuda')

embed_ref_module = LabelEmbedderRef(num_classes, hidden_size, dropout_prob).to('cuda')
with torch.no_grad():
    embed_ref_module.embedding_table.weight.copy_(embedding_table)
out_ref = embed_ref_module.forward(labels, train=False, force_drop_ids=torch.zeros_like(labels))
out_cuda = label_embed.label_embed_forward(labels.int(), embedding_table, force_drop)
assert torch.allclose(out_cuda, out_ref, atol=1e-5), "Label embedding kernel failed"
print("Label embedding kernel test passed.")
