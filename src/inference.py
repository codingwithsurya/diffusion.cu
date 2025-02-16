import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import numpy as np
from PIL import Image
import os
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
import urllib.request

# Pre-trained model URLs
MODEL_URLS = {
    '256': 'https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt',
    '512': 'https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt'
}

# Load CUDA modules
dit_block_mod = load(
    name="dit_block_mod",
    sources=[
        "src/cuda/dit_block.cu",
        "src/cuda/cuda_adaln_modulation.cu",
        "src/cuda/cuda_attention.cu",
        "src/cuda/cuda_mlp.cu",
        "src/cuda/cuda_timestep_embed.cu",
        "src/cuda/cuda_util_kernels.cu",
        "src/cuda/cuda_layernorm.cu",
        "src/cuda/cuda_label_embed.cu",
        "src/cuda/final_layer.cu"
    ],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

class DiTInference(nn.Module):
    def __init__(
        self,
        input_size=32,  # 256/8 = 32 for 256x256 images
        hidden_size=1152,  # DiT-XL/2 default
        depth=28,  # DiT-XL/2 default
        num_heads=16,  # DiT-XL/2 default
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=True
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        
        # Input embedding
        self.x_embedder = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_size ** 2, hidden_size))

        # Initialize weights for DiT blocks (loaded from checkpoint later)
        self.blocks = []
        for i in range(depth):
            block_params = {
                'hidden_size': hidden_size,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'qkv_bias': True
            }
            self.blocks.append(block_params)

        # Final layer norm and output projection
        self.final_layer = nn.Linear(hidden_size, 8 if learn_sigma else 4)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x):
        """Convert the latent patches back into image space."""
        h = w = int(np.sqrt(x.shape[1]))
        return x.reshape(shape=(x.shape[0], h, w, self.hidden_size))

    def forward(self, x, timesteps, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations)
        timesteps: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # 1. Input embedding and positioning
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)  # (N, T, C)
        x = x + self.pos_embed

        # 2. Time and class embeddings
        t_emb = dit_block_mod.timestep_embedding(timesteps, self.hidden_size)
        y_emb = dit_block_mod.label_embedding(y, self.num_classes, self.hidden_size)
        c = t_emb + y_emb

        # 3. DiT blocks
        for block_params in self.blocks:
            x = dit_block_mod.dit_block_forward(
                x, c,
                block_params['w_mod'], block_params['b_mod'],
                block_params['wQ'], block_params['bQ'],
                block_params['wK'], block_params['bK'],
                block_params['wV'], block_params['bV'],
                block_params['wO'], block_params['bO'],
                block_params['w1'], block_params['b1'],
                block_params['w2'], block_params['b2'],
                eps=1e-6
            )

        # 4. Final layer norm and output projection
        x = self.unpatchify(x)
        x = dit_block_mod.final_layer_forward(x)
        
        # 5. Split output into mean and variance if learning sigma
        if self.learn_sigma:
            x, log_var = x.chunk(2, dim=-1)
            return x, log_var
        return x, None

def download_model(image_size):
    """Download pretrained DiT model if not exists"""
    os.makedirs('pretrained', exist_ok=True)
    model_path = f'pretrained/DiT-XL-2-{image_size}x{image_size}.pt'
    
    if not os.path.exists(model_path):
        print(f"Downloading DiT-XL/2 {image_size}x{image_size} model...")
        urllib.request.urlretrieve(MODEL_URLS[str(image_size)], model_path)
    
    return model_path

def generate_images(
    prompt=None,  # Text prompt for class-conditional generation
    num_samples=1,
    image_size=256,  # Choose 256 or 512
    guidance_scale=7.5,
    num_inference_steps=50,
    seed=None,
    device="cuda"
):
    """
    Generate images using the CUDA-optimized DiT model.
    
    Args:
        prompt (str or int, optional): Text prompt or ImageNet class index (0-999)
        num_samples (int): Number of images to generate
        image_size (int): Output image size (256 or 512)
        guidance_scale (float): Classifier-free guidance scale
        num_inference_steps (int): Number of diffusion steps
        seed (int, optional): Random seed for reproducibility
        device (str): Device to run on ('cuda' or 'cpu')
    
    Returns:
        list[PIL.Image]: Generated images
    """
    assert image_size in [256, 512], "Image size must be either 256 or 512"
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        
    # Download and load pretrained model
    model_path = download_model(image_size)
    model = DiTInference(input_size=image_size//8)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    model.eval()
    
    # Convert prompt to class label if needed
    if isinstance(prompt, str):
        # TODO: Add text-to-class mapping for ImageNet
        class_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    elif isinstance(prompt, int):
        class_labels = torch.full((num_samples,), prompt, dtype=torch.long, device=device)
    else:
        class_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    
    # Create diffusion scheduler and VAE
    diffusion = create_diffusion(str(num_inference_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    
    # Sample latents and generate images
    with torch.no_grad():
        # Initialize latents
        latents = torch.randn(num_samples, 4, image_size//8, image_size//8, device=device)
        
        # Sampling function with classifier-free guidance
        def model_fn(x_t, t, y):
            if guidance_scale == 1:
                return model(x_t, t, y)[0]
            
            # For classifier-free guidance, run both conditional and unconditional forward passes
            x_in = torch.cat([x_t] * 2)
            t_in = torch.cat([t] * 2)
            y_in = torch.cat([y, torch.zeros_like(y)])
            
            noise_pred, _ = model(x_in, t_in, y_in)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Run diffusion sampling
        latents = diffusion.sample(model_fn, latents, class_labels)
        
        # Decode latents to images
        x = vae.decode(latents / 0.18215).sample
        x = torch.clamp((x + 1) / 2, 0, 1)
        x = (x * 255).round().to(torch.uint8)
        images = x.cpu().permute(0, 2, 3, 1).numpy()
        
        # Convert to PIL images
        return [Image.fromarray(img) for img in images]

if __name__ == "__main__":
    # Example usage
    images = generate_images(
        prompt="A photo of a cat",  # or use class index: prompt=281 for 'tabby cat'
        num_samples=4,
        image_size=256,
        guidance_scale=7.5
    )
    
    # Save generated images
    os.makedirs('outputs', exist_ok=True)
    for i, image in enumerate(images):
        image.save(f'outputs/sample_{i}.png')
