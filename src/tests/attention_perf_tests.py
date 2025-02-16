import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import time
import numpy as np

cuda_attention = load(
    name="cuda_attention",
    sources=["cuda_attention.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

def check_correctness(Q, K, V):
    """Verify CUDA implementation matches PyTorch reference"""
    print("\nVerifying Implementation Correctness")
    print("-" * 50)
    
    # Get outputs from both implementations
    out_cuda = cuda_attention.attention_forward(Q, K, V)
    out_ref = torch.matmul(
        F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1)), dim=-1),
        V
    )
    
    # Check error metrics
    max_error = (out_cuda - out_ref).abs().max().item()
    mean_error = (out_cuda - out_ref).abs().mean().item()
    
    print(f"Max absolute error:  {max_error:.6f}")
    print(f"Mean absolute error: {mean_error:.6f}")
    
    if torch.allclose(out_cuda, out_ref, atol=1e-5):
        print("✓ Implementation matches reference")
        return True
    else:
        print("✗ Implementation does not match reference!")
        return False

def benchmark_attention(batch_size=1, seq_len=32, dim=64, trials=5, warmup=100, iters=1000):
    """Benchmark attention implementation focusing on optimal configuration"""
    print("\nAttention Performance Analysis")
    print("=" * 50)
    print(f"Configuration: {batch_size}x{seq_len}x{dim}")
    
    # Create tensors
    torch.manual_seed(42)  # For reproducibility
    Q = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, dim, device='cuda', dtype=torch.float32)
    
    # First verify correctness
    if not check_correctness(Q, K, V):
        print("\nSkipping benchmark due to implementation mismatch!")
        return
    
    # Extensive warmup
    print(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = cuda_attention.attention_forward(Q, K, V)
        _ = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(dim), dim=-1), V)
    torch.cuda.synchronize()
    
    # Multiple trials
    cuda_times = []
    torch_times = []
    
    print(f"Running {trials} trials, {iters} iterations each...")
    for trial in range(trials):
        # CUDA implementation timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = cuda_attention.attention_forward(Q, K, V)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / iters * 1000  # ms
        cuda_times.append(cuda_time)
        
        # PyTorch reference timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(dim), dim=-1), V)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / iters * 1000  # ms
        torch_times.append(torch_time)
    
    # Get best results (lowest latency)
    best_cuda = min(cuda_times)
    best_torch = min(torch_times)
    
    # Calculate metrics
    speedup = best_torch / best_cuda
    improvement = (best_torch - best_cuda) / best_torch * 100
    tokens_per_sec_cuda = batch_size * seq_len / (best_cuda / 1000)
    tokens_per_sec_torch = batch_size * seq_len / (best_torch / 1000)
    
    print("\nPerformance Results")
    print("-" * 50)
    print(f"Best Latency (over {trials} trials):")
    print(f"  CUDA Implementation:  {best_cuda:.3f} ms")
    print(f"  PyTorch Reference:    {best_torch:.3f} ms")
    print(f"  Speedup:             {speedup:.2f}x")
    print(f"  Performance Gain:    {improvement:.1f}%")
    
    print(f"\nThroughput:")
    print(f"  CUDA Implementation:  {tokens_per_sec_cuda/1000:.1f}k tokens/sec")
    print(f"  PyTorch Reference:    {tokens_per_sec_torch/1000:.1f}k tokens/sec")
    print(f"  Throughput Ratio:    {tokens_per_sec_cuda/tokens_per_sec_torch:.2f}x")

if __name__ == "__main__":
    # Run with optimal configuration
    benchmark_attention()