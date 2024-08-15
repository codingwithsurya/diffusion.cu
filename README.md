# diffusion.cu

This project is a from-scratch implementation of a UNet for diffusion model training in C++/CUDA. Inspired by Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c), it aims to understand diffusion models and CUDA programming while striving for performance comparable to PyTorch. The implementation is based on the research paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).

**My Motivation:**

As a Python programmer, I was fascinated by diffusion models but found the math and implementation details challenging. I also wanted to delve into the world of CUDA and understand how to leverage its power for faster deep learning. This project was born out of my desire to learn by doing, and to see if I could achieve performance comparable to, or even exceeding, PyTorch. Python can be slow, especially for computationally intensive tasks like training diffusion models, so the appeal of C++/CUDA's speed was undeniable.

**Key Features:**

* **Pure C++/CUDA Implementation:** The entire UNet, including the training loop, is written in C++/CUDA for maximum performance.
* **Unconditional Diffusion Training:**  Currently supports unconditional diffusion model training.
* **Performance Benchmarks:** Includes comparisons against PyTorch with and without `torch.compile` on an RTX 4090.

**My Goal: Beating `torch.compile`**

One of my ambitious goals for this project is to try and outperform PyTorch's `torch.compile` feature. `torch.compile` is a game-changer for PyTorch performance. It works by taking your PyTorch model and optimizing it under the hood using techniques like graph compilation, operator fusion, and more. This results in significantly faster execution, especially on NVIDIA GPUs. It's a tough challenge, but I'm excited to see how close I can get!

**Current Performance:**

The project is still a work in progress, but it's showing promising results! The end-to-end training loop is currently running at about 40% the speed of PyTorch with `torch.compile`.

**Learning Resources That Helped Me:**

If you're interested in learning more about diffusion models and CUDA programming, here are some resources that I found incredibly helpful:

* **Understanding Diffusion Models:**
    * **Research Paper Deep Dive (Highly Recommended):** [https://www.youtube.com/watch?v=W-O7AZNzbzQ](https://www.youtube.com/watch?v=W-O7AZNzbzQ) - This video provides a great explanation of the research paper.
    * **Demystifying the Math:** [https://www.youtube.com/watch?v=HoKDTa5jHvg](https://www.youtube.com/watch?v=HoKDTa5jHvg) - If you're struggling with the math behind diffusion models, like I was, this video is a lifesaver.

* **My Journey into CUDA:**
    * **Programming Massively Parallel Processors (Book & Lecture Series):** [https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) - This is a great starting point for learning the fundamentals of CUDA.
    * **Getting Started with CUDA for Python Programmers:** [https://www.youtube.com/watch?v=nOxKexn3iBo](https://www.youtube.com/watch?v=nOxKexn3iBo) - A fantastic introductory YouTube series specifically for Python programmers venturing into CUDA.
    * **My Optimization Bible: CUDA Matrix Multiplication Optimization Tutorial:** [https://siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM) - This tutorial is where I learned the majority of the optimization techniques I used in this project. Highly recommended!

**More CUDA/GPU Programming Resources:**

I've compiled a more extensive list of valuable resources in the repository's full README.

**Acknowledgments:**

Huge thanks to Andrej Karpathy for his inspiring [llm.c](https://github.com/karpathy/llm.c) project and to the authors of the research paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233). Also, credit to [clu0/unet.cu](https://github.com/clu0/unet.cu) and [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM) for providing valuable code inspiration.