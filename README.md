# diffusion.cu

This project is a from-scratch implementation of diffusion model training in C++/CUDA. Inspired by Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c). The implementation is based on the U-Net architecture in the paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).

**My Motivation:**

As a Python programmer, I was fascinated by diffusion models but found the math and implementation details challenging. Meanwhile, because of my interest in ML systems and infrastructure, I also wanted to learn CUDA, and understand how to get the most out of GPUs. This project was born out of my desire to learn by doing, and to see if I could achieve performance comparable to, or even exceeding, PyTorch. Python can be slow, especially for computationally intensive tasks like training diffusion models, so the appeal of C++/CUDA's speed was undeniable.

**My Goal: Beating `torch.compile`**

One of the primary objectives for this project is to develop a solution that can potentially surpass the performance of PyTorch's torch.compile feature. torch.compile leverages advanced optimization techniques such as just-in-time (JIT) graph compilation, operator fusion, and low-level kernel optimizations to enhance the execution efficiency of PyTorch models, particularly on NVIDIA GPUs. These optimizations significantly improve runtime performance by reducing overhead and maximizing hardware resource utilization. In fact, PyTorch runs heuristics directly on your hardware to squeeze out every bit of performance. This results in significantly faster execution, especially on NVIDIA GPUs. It's a tough challenge, but I'm excited to see how close I can get!

**Current Implementation:**

This currently supports unconditional diffusion model training, and the end-to-end training loop is currently running at about 55% the speed of PyTorch with `torch.compile` when run on a single H100. Our main bottleneck is memory bandwidth saturation during shared memory loads for convolutions, but we can potentially optimize it by tweaking tiling, exploring register blocking, and mayb e even leveraging H100's Transformer Engine and FP8 precision.

**Learning Resources That Helped Me:**

If you're interested in learning more about diffusion models and CUDA programming, here are some resources that I found incredibly helpful:

* **Understanding Diffusion Models:**
    [https://www.youtube.com/watch?v=W-O7AZNzbzQ](https://www.youtube.com/watch?v=W-O7AZNzbzQ) - This video provides a great explanation of the research paper.
    [https://www.youtube.com/watch?v=HoKDTa5jHvg](https://www.youtube.com/watch?v=HoKDTa5jHvg) - If you're struggling with the math behind diffusion models, like I was, this video is a lifesaver.

* **My Journey into CUDA:**
    * **Programming Massively Parallel Processors (Book & Lecture Series):** [https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) - This is a great starting point for learning the fundamentals of GPU Programming/HPC. 
    * **Getting Started with CUDA for Python Programmers:** [https://www.youtube.com/watch?v=nOxKexn3iBo](https://www.youtube.com/watch?v=nOxKexn3iBo) - Great introductory YouTube series specifically for Python programmers venturing into CUDA.
    * **My Optimization Bible: CUDA Matrix Multiplication Optimization Tutorial:** [https://siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM) - This tutorial is where I learned the majority of the optimization techniques I used in this project. Highly recommended!

**More CUDA/GPU Programming Resources:**

## High Quality Resources on GPU Programming/Architecture  

### Articles/Blogs

- [GPU Programming](https://enccs.github.io/gpu-programming/)
- [The CUDA Parallel Programming Model](https://fabiensanglard.net/cuda/)
- [A HISTORY OF NVIDIA STREAM MULTIPROCESSOR](https://fabiensanglard.net/cuda/index.html)
- [Parallel Thread Execution](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [What Every Developer Should Know About GPU Computing](https://codeconfessions.substack.com/p/gpu-computing)
- [A minimal GPU design in Verilog to learn how GPUs work from the ground up](https://github.com/adam-maj/tiny-gpu)
- [GPU Programming: When, Why and How?](https://enccs.github.io/gpu-programming/)
- [Understanding GPU internals](https://cmeraki.github.io/gpu-part1.html)
- [Understanding the GPU programming model](https://cmeraki.github.io/gpu-part2.html)
  
### Tutorials 
- [Intro to Parallel Programming](https://developer.nvidia.com/udacity-cs344-intro-parallel-programming)

### Notebooks
- [GPU Puzzles](https://github.com/srush/GPU-Puzzles)
  
### Videos 
- [How GPU Computing Works](https://www.youtube.com/watch?v=3l10o0DYJXg)
- [Getting Started With CUDA for Python Programmers](https://youtu.be/nOxKexn3iBo?si=nung2_X-TXsnK4YK)
- [Programming Massively Parallel Processors - Lecture Series by the Book Author](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4)
- [Programming Massively Parallel Processors: A Hands-on Approach and then this YT series](https://m.youtube.com/playlist?list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj&si=ZqKCQgFef-v3JBv8)
- [Programming Parallel Computers](https://youtube.com/playlist?list=PL2RY7P3JxZN-Pz1nwvnoJ9uEHmOmv4jmi&si=-7hc_4fQfFrMc8VZ)
- [GPU Programming Lectures](https://youtube.com/playlist?list=PL3xCBlatwrsXCGW4SfEoLzKiMSUCE7S_X&si=2vIw6R0JpZjBt8pR)
- [From Scratch CUDA](https://youtube.com/playlist?list=PLxNPSjHT5qvvwoy6KXzUbLaF5A8NdJvuo&si=rvc52nc-VAPVwhNh)
- [CUDA Programming](https://www.youtube.com/watch?v=xwbD6fL5qC8)
- [CUDA MODE Lectures](https://www.youtube.com/@CUDAMODE/videos)


**Acknowledgments:**

Huge thanks to Andrej Karpathy for his inspiring [llm.c](https://github.com/karpathy/llm.c) project and to the authors of the research paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233). Also, credit to [clu0/unet.cu](https://github.com/clu0/unet.cu) and [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM) for providing valuable code inspiration.
