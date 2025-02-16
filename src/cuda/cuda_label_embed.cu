// cuda_label_embed.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

//---------------------------------------------------------
// Label Embedding Kernel
//
// For each sample n and each embedding dimension j:
//   If force_drop[n]==1 then label = num_classes (the extra embedding)
//   Otherwise, label = labels[n].
// Then, out[n, j] = embedding_table[label, j].
// 
// labels: [N] (int32)
// force_drop: [N] (int32)  (if not used, pass an array of 0’s)
// embedding_table: [num_classes+1, hidden_size] (float)
// out: [N, hidden_size] (float)
//
__global__ void label_embed_kernel(const int* __restrict__ labels,
                                   const int* __restrict__ force_drop,
                                   const float* __restrict__ embedding_table,
                                   float* __restrict__ out,
                                   int N, int hidden_size, int num_classes) {
    int n = blockIdx.x * blockDim.x + threadIdx.x; // over batch
    int j = blockIdx.y * blockDim.y + threadIdx.y;   // over embedding dim
    if (n < N && j < hidden_size) {
        int label = labels[n];
        if (force_drop[n] == 1)
            label = num_classes;  // Use the extra embedding index
        out[n * hidden_size + j] = embedding_table[label * hidden_size + j];
    }
}

//---------------------------------------------------------
// Wrapper for label embedding.
// If force_drop is not provided, caller can pass a tensor of zeros.
torch::Tensor label_embed_forward(torch::Tensor labels, torch::Tensor embedding_table, torch::Tensor force_drop) {
    TORCH_CHECK(labels.is_cuda(), "labels must be CUDA tensor");
    TORCH_CHECK(embedding_table.is_cuda(), "embedding_table must be CUDA tensor");
    TORCH_CHECK(force_drop.is_cuda(), "force_drop must be CUDA tensor");
    int N = labels.size(0);
    int hidden_size = embedding_table.size(1);
    // num_classes is (num_rows - 1) since the extra row is used for dropout.
    int num_classes = embedding_table.size(0) - 1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto out = torch::empty({N, hidden_size}, options);
    dim3 blockDim(32, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (hidden_size + blockDim.y - 1) / blockDim.y);
    label_embed_kernel<<<gridDim, blockDim>>>(labels.data_ptr<int>(),
                                              force_drop.data_ptr<int>(),
                                              embedding_table.data_ptr<float>(),
                                              out.data_ptr<float>(),
                                              N, hidden_size, num_classes);
    cudaDeviceSynchronize();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("label_embed_forward", &label_embed_forward, "Label Embedding forward (CUDA)");
}