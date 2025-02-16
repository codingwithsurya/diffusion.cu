import torch
import numpy as np

# --- Test 1: modulate_forward ---
# PyTorch reference: out = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
def test_modulate():
    # Create random tensors
    N, T, D = 2, 64, 128
    x = torch.rand(N, T, D, device='cuda')
    shift = torch.full((N, D), 0.1, device='cuda')
    scale = torch.full((N, D), 0.2, device='cuda')
    
    # PyTorch reference implementation
    expected = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    # Call your CUDA extension
    output = cuda_util.modulate_forward(x, shift, scale)
    
    # Compare (using a tolerance)
    assert torch.allclose(output, expected, atol=1e-5), "modulate_forward test failed"
    print("modulate_forward test passed.")

# --- Test 2: sincos_pos_embed_forward ---
# PyTorch reference: We'll re-implement the reference (as in models.py) in Python.
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    # Create grid of positions
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # note: x then y
    grid = np.stack(grid, axis=0)  # shape [2, grid_size, grid_size]
    grid = grid.reshape(2, 1, grid_size, grid_size)
    
    # We'll compute the embedding similarly to the CUDA kernel.
    num_positions = grid_size * grid_size
    pos_embed = np.empty((num_positions, embed_dim), dtype=np.float32)
    half = embed_dim // 2
    quarter = half // 2
    for idx in range(num_positions):
        i = idx // grid_size
        j = idx % grid_size
        emb = np.empty(embed_dim, dtype=np.float32)
        for k in range(quarter):
            omega = np.exp(-np.log(10000) * (k / half))
            val = i * omega
            emb[k] = np.sin(val)
            emb[k + quarter] = np.cos(val)
        for k in range(quarter):
            omega = np.exp(-np.log(10000) * (k / half))
            val = j * omega
            emb[half + k] = np.sin(val)
            emb[half + k + quarter] = np.cos(val)
        pos_embed[idx, :] = emb
    return pos_embed

def test_sincos_pos_embed():
    grid_size = 16
    embed_dim = 128
    # Get CUDA output
    pos_embed_cuda = cuda_util.sincos_pos_embed_forward(embed_dim, grid_size)
    pos_embed_cuda_np = pos_embed_cuda.cpu().numpy()
    
    # PyTorch reference output
    pos_embed_ref = get_2d_sincos_pos_embed(embed_dim, grid_size)
    
    # Compare
    assert np.allclose(pos_embed_cuda_np, pos_embed_ref, atol=1e-5), "sincos_pos_embed_forward test failed"
    print("sincos_pos_embed_forward test passed.")

# Run tests
test_modulate()
test_sincos_pos_embed()
