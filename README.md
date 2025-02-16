# diffusion.cu

High-performance Diffusion Transformer (DiT) implementation from scratch using CUDA/C++. Features optimized CUDA kernels with:

### MLP Block
- Matrix multiplications using shared memory and warp-level tiling
- Persistent threadblocks for efficient kernel reuse
- Tensor Core acceleration via WMMA API
- Kernel fusion for SiLU activation and bias addition
- Mixed precision (FP16) computation

### Attention Block
- Optimized scaled dot-product attention using shared memory tiling
- Fused softmax with numerical stability optimizations
- Efficient parallel reductions for attention scores
- Block-level parallelism for multi-head attention
- Memory coalescing for Q, K, V matrix operations

## Usage

```python
from src.inference import generate_images

images = generate_images(
    prompt="A photo of a cat",  # or class index (0-999)
    image_size=256,            # or 512
    num_samples=4
)
```

Automatically downloads and runs inference using pretrained DiT-XL/2 models.

## Requirements
- CUDA 11.0+
- PyTorch 2.0+
- NVIDIA GPU with Tensor Cores

## Acknowledgments

This implementation is based on:
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) by William Peebles and Saining Xie
- [Official PyTorch DiT Implementation](https://github.com/facebookresearch/DiT) by Facebook Research, used for benchmarking and validation

The CUDA kernels in this repository are written from scratch but validated against the official implementation to ensure correctness.
