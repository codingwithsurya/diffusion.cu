#include "label_embedder.cuh"
#include "utils.cuh"

// ----------------------------------------------------------------------------
// Label Embedder

void label_embedder_forward(
    LabelEmbedder *l_embedder,
    const int *labels,
    float dropout_prob,
    int block_size)
{
    // similar to embedding layer in llm.c

    int B = l_embedder->B;
    int hidden_size = l_embedder->hidden_size;
    const float *embedding_table = l_embedder->embedding_table;

    // during training, replace label with num_classes with probability dropout_prob
    if (dropout_prob > 0)
    {
        // generate random numbers to decide which labels to drop
        float *rand_nums;
        cudaMalloc(&rand_nums, B * sizeof(float));
        curandGenerateUniform(l_embedder->curand_states, rand_nums, B);
        // replace label with num_classes if random number is less than dropout_prob
        int *labels_device;
        cudaMalloc(&labels_device, B * sizeof(int));
        cudaMemcpy(labels_device, labels, B * sizeof(int), cudaMemcpyHostToDevice);
        int num_classes = l_embedder->num_classes;
        for (int i = 0; i < B; i++)
        {
            if (rand_nums[i] < dropout_prob)
            {
                labels_device[i] = num_classes;
            }
        }
        cudaFree(rand_nums);
        // now copy to output
        for (int b = 0; b < B; b++)
        {
            int label = labels_device[b];
            for (int i = 0; i < hidden_size; i++)
            {
                l_embedder->out[b * hidden_size + i] = embedding_table[label * hidden_size + i];
            }
        }
        cudaFree(labels_device);
    }
    else
    {
        // if dropout_prob is 0, simply copy the embeddings from the table
        for (int b = 0; b < B; b++)
        {
            int label = labels[b];
            for (int i = 0; i < hidden_size; i++)
            {
                l_embedder->out[b * hidden_size + i] = embedding_table[label * hidden_size + i];
            }
        }
    }
}

void label_embedder_backward(
    LabelEmbedder *l_embedder,
    const float *dout,
    int block_size)
{
    int B = l_embedder->B;
    int hidden_size = l_embedder->hidden_size;
    float *embedding_table = l_embedder->embedding_table;
    // simply add the gradient to the embedding table, no need for special logic
    // because each label is only used once per batch
    add_inplace_forward(dout, embedding_table, B * hidden_size, block_size);
}

void label_embedder_init(LabelEmbedder *l_embedder, int num_classes, int hidden_size, int B, int block_size, unsigned long long seed)
{
    // init random number generator
    cudaCheck(cudaMalloc(&(l_embedder->curand_states), B * sizeof(curandState)));
    init_curand_states<<<ceil_div(B, block_size), block_size>>>(l_embedder->curand_states, seed, B);
    cudaCheck(cudaGetLastError());

    l_embedder->num_classes = num_classes;
    l_embedder->hidden_size = hidden_size;
    l_embedder->B = B;
    l_embedder->block_size = block_size;

    // allocate memory for embedding table
    cudaCheck(cudaMalloc(&(l_embedder->embedding_table), (num_classes + 1) * hidden_size * sizeof(float)));
    cudaCheck(cudaMemset(l_embedder->embedding_table, 0, (num_classes + 1) * hidden_size * sizeof(float)));

    // allocate memory for output
    cudaCheck(cudaMalloc(&(l_embedder->out), B * hidden_size * sizeof(float)));
}

void label_embedder_free(LabelEmbedder *l_embedder)
{
    cudaCheck(cudaFree(l_embedder->curand_states));
    cudaCheck(cudaFree(l_embedder->embedding_table));
    cudaCheck(cudaFree(l_embedder->out));
}