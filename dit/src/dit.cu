#include "dit.h"
#include "utils.cuh"
#include "patch_embed.cuh"
#include "timestep_embedder.cuh"
#include "label_embedder.cuh"
#include "dit_block.cuh"
#include "final_layer.cuh"
#include <assert.h>

void dit_forward(
    DiT *dit,
    const float *x, // Input image: (B, C, H, W)
    const float *t, // Timesteps: (B,)
    const int *y,   // Class labels: (B,)
    cublasHandle_t cublas_handle,
    float cfg_scale, // Classifier-free guidance scale
    bool use_cfg,    // Whether to use classifier-free guidance
    float *t_out = nullptr)
{
    // Constants
    int B = dit->config.B;
    int C = dit->config.C;
    int H = dit->config.H;
    int W = dit->config.W;
    int patch_size = dit->config.patch_size;
    int hidden_size = dit->config.hidden_size;
    int dim = dit->config.dim;
    int num_classes = dit->config.num_classes;
    int block_size = dit->config.block_size;
    int depth = dit->config.depth;
    float dropout_prob = dit->config.dropout_prob;

    // --- Input Processing ---

    // 1. Embed the input image into patches
    embed_patches_forward(x, dit->patch_embeddings, B, C, H, W, patch_size, block_size);

    // 2. Get timestep embeddings
    timestep_embedder_forward(&dit->t_embedder, t, cublas_handle);

    // 3. Get label embeddings (with dropout during training)
    label_embedder_forward(&dit->l_embedder, y, dropout_prob, block_size);

    // 4. Add timestep and label embeddings
    add_forward(dit->t_embedder.out, dit->l_embedder.out, dit->t_plus_y, B * hidden_size, block_size);

    // --- Transformer Blocks ---
    float *block_out = dit->patch_embeddings; // Output of the previous block
    cudaEvent_t start, end;
    float total_time = 0.0f;
    if (t_out != nullptr)
    {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
    }

    for (int l = 0; l < depth; l++)
    {
        if (t_out != nullptr)
        {
            cudaCheck(cudaEventRecord(start));
        }
        dit_block_forward(&dit->blocks[l], block_out, dit->t_plus_y, cublas_handle, B, dit->patch_embedder.N, hidden_size, block_size);
        if (t_out != nullptr)
        {
            cudaCheck(cudaEventRecord(end));
            cudaCheck(cudaEventSynchronize(start));
            cudaCheck(cudaEventSynchronize(end));
            float elapsed_time;
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
            total_time += elapsed_time;
            printf("block %d: time: %f\n", l, elapsed_time);
        }
        block_out = dit->blocks[l].attn_out; // Update output for the next block
    }
    if (t_out != nullptr)
    {
        *t_out += total_time;
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(end));
    }

    // --- Classifier-Free Guidance (CFG) ---
    if (use_cfg)
    {
        assert(cfg_scale >= 1.0f);

        // Split the input and conditioning into two halves for conditional and unconditional paths
        int half_B = B / 2;
        float *x_cond = dit->patch_embeddings;
        float *x_uncond = dit->patch_embeddings + half_B * dit->patch_embedder.N * hidden_size;
        float *c_cond = dit->t_plus_y;
        float *c_uncond = dit->t_plus_y + half_B * hidden_size;

        // Forward pass for both conditional and unconditional paths
        dit_forward(dit, x_cond, t, y, cublas_handle, cfg_scale, false, t_out);         // Conditional path (already done above)
        dit_forward(dit, x_uncond, t, nullptr, cublas_handle, cfg_scale, false, t_out); // Unconditional path

        // Apply CFG: Interpolate between conditional and unconditional outputs
        float *out_cond = dit->final_layer.out;
        float *out_uncond = dit->final_layer.out + half_B * dit->patch_embedder.N * dit->final_layer.out_channels * patch_size * patch_size;
        for (int i = 0; i < half_B * dit->patch_embedder.N * dit->final_layer.out_channels * patch_size * patch_size; i++)
        {
            out_cond[i] = (1 - cfg_scale) * out_uncond[i] + cfg_scale * out_cond[i];
        }
    }

    // --- Final Layer ---
    final_layer_forward(&dit->final_layer, block_out, dit->t_plus_y, cublas_handle);
}

void dit_backward(
    DiT *dit,
    const float *dout,
    const float *x,
    const float *t,
    const int *y,
    cublasHandle_t cublas_handle,
    float cfg_scale,
    bool use_cfg,
    float *t_out = nullptr)
{
    int B = dit->config.B;
    int C = dit->config.C;
    int H = dit->config.H;
    int W = dit->config.W;
    int patch_size = dit->config.patch_size;
    int hidden_size = dit->config.hidden_size;
    int dim = dit->config.dim;
    int depth = dit->config.depth;
    int block_size = dit->config.block_size;

    cudaEvent_t start, end;
    float total_time = 0.0f;
    if (t_out != nullptr)
    {
        cudaCheck(cudaEventCreate(&start));
        cudaCheck(cudaEventCreate(&end));
    }
    // --- Backward through Final Layer ---
    final_layer_backward(&dit->final_layer, dout, dit->blocks[depth - 1].attn_out, dit->t_plus_y, cublas_handle);

    // --- Backward through Transformer Blocks ---
    float *block_dout = dit->final_layer.dx; // Gradient from the next block (final layer in this case)
    for (int l = depth - 1; l >= 0; l--)
    {
        if (t_out != nullptr)
        {
            cudaCheck(cudaEventRecord(start));
        }
        dit_block_backward(&dit->blocks[l], block_dout, dit->t_plus_y, cublas_handle, B, dit->patch_embedder.N, hidden_size, block_size);
        if (t_out != nullptr)
        {
            cudaCheck(cudaEventRecord(end));
            cudaCheck(cudaEventSynchronize(start));
            cudaCheck(cudaEventSynchronize(end));
            float elapsed_time;
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, end));
            total_time += elapsed_time;
            printf("block %d: time: %f\n", l, elapsed_time);
        }
        block_dout = dit->blocks[l].dx; // Update gradient for the previous block
    }

    if (t_out != nullptr)
    {
        *t_out += total_time;
        cudaCheck(cudaEventDestroy(start));
        cudaCheck(cudaEventDestroy(end));
    }

    // --- Classifier-Free Guidance (CFG) ---
    if (use_cfg)
    {
        assert(cfg_scale >= 1.0f);

        int half_B = B / 2;

        // Split the output gradient into two halves for conditional and unconditional paths
        float *dout_cond = dout;
        float *dout_uncond = dout + half_B * dit->final_layer.N * dit->final_layer.out_channels * patch_size * patch_size;

        // Apply CFG scaling to the output gradients
        for (int i = 0; i < half_B * dit->final_layer.N * dit->final_layer.out_channels * patch_size * patch_size; i++)
        {
            dout_cond[i] *= cfg_scale;
            dout_uncond[i] *= (1 - cfg_scale);
        }

        // Split the input and conditioning into two halves for conditional and unconditional paths
        float *x_cond = x;
        float *x_uncond = x + half_B * C * H * W;
        float *t_cond = t;
        float *t_uncond = t + half_B;

        // Backward pass for both conditional and unconditional paths
        dit_backward(dit, dout_cond, x_cond, t_cond, y, cublas_handle, 1.0f, false, t_out);             // Conditional path
        dit_backward(dit, dout_uncond, x_uncond, t_uncond, nullptr, cublas_handle, 1.0f, false, t_out); // Unconditional path

        // Combine gradients for input image
        for (int i = 0; i < half_B * C * H * W; i++)
        {
            dit->dx[i] = cfg_scale * dit->dx[i] + (1 - cfg_scale) * dit->dx[i + half_B * C * H * W];
        }

        // Combine gradients for timestep embeddings
        for (int i = 0; i < half_B * hidden_size; i++)
        {
            dit->t_embedder.out[i] = cfg_scale * dit->t_embedder.out[i] + (1 - cfg_scale) * dit->t_embedder.out[i + half_B * hidden_size];
        }

        // Combine gradients for label embeddings
        for (int i = 0; i < half_B * hidden_size; i++)
        {
            dit->l_embedder.out[i] = cfg_scale * dit->l_embedder.out[i] + (1 - cfg_scale) * dit->l_embedder.out[i + half_B * hidden_size];
        }
    }

    // --- Backward through Input Processing ---

    // 4. Add timestep and label embedding gradients
    add_forward(dit->final_layer.dc, block_dout, dit->dt_plus_dy, B * hidden_size, block_size);

    // 3. Backward through label embedding
    label_embedder_backward(&dit->l_embedder, dit->dt_plus_dy, block_size);

    // 2. Backward through timestep embedding
    timestep_embedder_backward(&dit->t_embedder, dit->dt_plus_dy, cublas_handle);

    // 1. Backward through patch embedding
    embed_patches_backward(dit->dt_plus_dy, dit->dx, B, C, H, W, patch_size, block_size);
}

// ----------------------------------------------------------------------------
// memory management

void dit_init(DiT *dit, const DiTConfig config)
{
    // init the config
    dit->config = config;
    int B = config.B;
    int C = config.C;
    int H = config.H;
    int W = config.W;
    int patch_size = config.patch_size;
    int hidden_size = config.hidden_size;
    int dim = config.dim;
    int depth = config.depth;
    int num_classes = config.num_classes;
    int block_size = config.block_size;

    // allocate memory
    cudaCheck(cudaMalloc(&(dit->patch_embeddings), B * hidden_size * dit->final_layer.N * sizeof(float)));
    cudaCheck(cudaMalloc(&(dit->t_plus_y), B * hidden_size * sizeof(float)));
    cudaCheck(cudaMalloc(&(dit->dt_plus_dy), B * hidden_size * sizeof(float)));

    dit->blocks = (DiTBlock *)mallocCheck(depth * sizeof(DiTBlock));
    for (int l = 0; l < depth; l++)
    {
        dit_block_init(&dit->blocks[l], hidden_size, hidden_size, B, dit->final_layer.N, block_size);
    }

    // init the embedders
    timestep_embedder_init(&(dit->t_embedder), dim, hidden_size, B, config.max_period, block_size);
    label_embedder_init(&(dit->l_embedder), num_classes, hidden_size, B, block_size, 1337);
    final_layer_init(&(dit->final_layer), patch_size, C, C, B, H, W, hidden_size, block_size);

    // set the input pointer for the first block
    dit->blocks[0].x = dit->patch_embeddings;
    // set the output pointer for the last block
    dit->blocks[depth - 1].attn_out = dit->final_layer.dx;
    // set the input pointer for the final layer
    dit->final_layer.fc.inp = dit->t_plus_y;
    // set the input pointer for the projection layer in final layer
    dit->final_layer.proj.inp = dit->final_layer.ln_out;
    // set the input pointer for the patch embedder
    dit->patch_embedder.img = x;

    // allocate memory for the output of the model
    cudaCheck(cudaMalloc(&(dit->out), B * C * H * W * sizeof(float)));
    // copy output from final layer
    cudaCheck(cudaMemcpy(dit->out, dit->final_layer.out, B * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaCheck(cudaGetLastError());
}

void dit_free(DiT *dit)
{
    cudaCheck(cudaFree(dit->patch_embeddings));
    cudaCheck(cudaFree(dit->t_plus_y));
    cudaCheck(cudaFree(dit->dt_plus_dy));
    cudaCheck(cudaFree(dit->out));

    for (int l = 0; l < dit->config.depth; l++)
    {
        dit_block_free(&dit->blocks[l]);
    }
    free(dit->blocks);
    timestep_embedder_free(&dit->t_embedder);
    label_embedder_free(&dit->l_embedder);
    final_layer_free(&dit->final_layer);
}