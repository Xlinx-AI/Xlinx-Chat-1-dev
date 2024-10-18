# ==========================================
# Install Necessary Libraries
# ==========================================

# Uncomment and run these lines if you're setting up a new environment.
# They ensure the correct versions of PyTorch and other dependencies are installed.

# !pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
# !pip install transformers datasets pillow fastapi uvicorn tiktoken einops tensorboard
# !pip install faiss-cpu slowapi tqdm

# ==========================================
# Imports and Device Initialization
# ==========================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import LongformerTokenizer, LongformerModel
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Callable
import uuid
from tqdm import tqdm
import argparse

# Import necessary modules for regularization and rate limiting
from fastapi import FastAPI, UploadFile, File, Form, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from collections import defaultdict
import threading

# ==========================================
# Utility Functions
# ==========================================

def initialize_weights(module: nn.Module):
    """Initialize weights for linear and normalization layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """Load model checkpoint."""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint_data = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        epoch = checkpoint_data['epoch']
        loss = checkpoint_data['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {epoch}, loss {loss})")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return None, None

# ==========================================
# Regularization Modules
# ==========================================

class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x / keep_prob * binary_mask

class DropBlock(nn.Module):
    """DropBlock regularization for spatial data."""
    def __init__(self, block_size: int = 7, drop_prob: float = 0.1):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand_like(x) < gamma).float()
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - (mask > 0).float()
        count = mask.numel() / mask.shape[0]
        return x * mask * (count / mask.sum())

class LayerDrop(nn.Module):
    """LayerDrop regularization for entire layers."""
    def __init__(self, drop_prob: float = 0.0):
        super(LayerDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, layer_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return layer_fn(x)
        if torch.rand(1).item() < self.drop_prob:
            return x
        return layer_fn(x)

# ==========================================
# Liquid Layers
# ==========================================

class LiquidLinear(nn.Module):
    """Dynamic Linear layer with adaptive weights."""
    def __init__(self, in_features: int, out_features: int, adapt_dim: int):
        super(LiquidLinear, self).__init__()
        self.base_linear = nn.Linear(in_features, out_features)
        self.adapt_linear = nn.Linear(adapt_dim, out_features * in_features)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        # Generate adaptive weights based on adapt_input
        adapt_weight = self.adapt_linear(adapt_input).view(self.base_linear.weight.size())
        # Combine base weights with adaptive weights
        weight = self.base_linear.weight + adapt_weight
        return F.linear(x, weight, self.base_linear.bias)

# ==========================================
# Vector Quantizer and VQVAE
# ==========================================

class VectorQuantizer(nn.Module):
    """Vector Quantizer for VQVAE."""
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # Flatten input
        flat_z = z.view(-1, self.embedding_dim)

        # Compute distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(flat_z, self.embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized,
            "vq_loss": loss,
            "perplexity": perplexity
        }

class VQVAE(nn.Module):
    """Vector Quantized Variational Autoencoder (VQVAE) for image tokenization."""
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.ReLU(),
            nn.Conv2d(256, embedding_dim, kernel_size=3, stride=1, padding=1)  # [B, embedding_dim, 32, 32]
        )

        # Vector Quantizer
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 128, 128]
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),  # [B, 3, 128, 128]
            nn.Sigmoid()
        )

        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor):
        """Forward pass through VQVAE."""
        z_e = self.encoder(x)  # Encode input
        vq_outputs = self.vq_layer(z_e)  # Vector quantization
        z_q = vq_outputs["quantized"]
        vq_loss = vq_outputs["vq_loss"]
        perplexity = vq_outputs["perplexity"]
        x_recon = self.decoder(z_q)  # Reconstruct input
        return {
            "quantized": z_q,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "reconstructed": x_recon
        }

# ==========================================
# Mixture of Experts Components
# ==========================================

class KolmogorovArnoldExpert(nn.Module):
    """Kolmogorov-Arnold Expert with non-linear activations."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, activation: str = 'gelu'):
        super(KolmogorovArnoldExpert, self).__init__()
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Each phi function processes one input dimension
        self.phi_functions = nn.ModuleList([nn.Sequential(
            nn.Linear(1, hidden_dim),
            act_fn
        ) for _ in range(input_dim)])
        # Psi function combines all phi outputs
        self.psi_function = nn.Sequential(
            nn.Linear(input_dim * hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Kolmogorov-Arnold expert."""
        # Apply each phi function to corresponding input dimension
        phi_outputs = [phi(x[:, i].unsqueeze(1)) for i, phi in enumerate(self.phi_functions)]
        # Concatenate all phi outputs
        concatenated = torch.cat(phi_outputs, dim=1)
        # Apply psi function
        return self.psi_function(concatenated)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts module with attention-based gating."""
    def __init__(
        self, 
        expert_dim: int, 
        num_experts: int, 
        adapt_dim: int, 
        hidden_dim: int = 64,
        drop_prob: float = 0.0,
        activation: str = 'gelu'
    ):
        super(MixtureOfExperts, self).__init__()
        # Create a list of LiquidLinear experts
        self.experts = nn.ModuleList([
            LiquidLinear(expert_dim, expert_dim, adapt_dim)
            for _ in range(num_experts)
        ])
        # Add Kolmogorov-Arnold expert
        self.ka_expert = KolmogorovArnoldExpert(expert_dim, expert_dim, hidden_dim, activation=activation)
        # Gating network to decide which expert to use
        self.gating = nn.Linear(adapt_dim, num_experts + 1)  # +1 for ka_expert
        self.drop_path = DropPath(drop_prob)
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through Mixture of Experts."""
        # Compute gating scores
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)  # [batch_size, num_experts + 1]
        expert_outputs = []
        # Compute outputs from each LiquidLinear expert
        for i, expert in enumerate(self.experts):
            expert_output = expert(x, adapt_input)  # [batch_size, expert_dim]
            expert_weight = gate_scores[:, i].unsqueeze(1)  # [batch_size, 1]
            expert_outputs.append(expert_weight * expert_output)
        # Compute output from Kolmogorov-Arnold expert
        ka_output = self.ka_expert(x)  # [batch_size, expert_dim]
        ka_weight = gate_scores[:, -1].unsqueeze(1)  # [batch_size, 1]
        expert_outputs.append(ka_weight * ka_output)
        # Sum all expert outputs
        output = sum(expert_outputs)  # [batch_size, expert_dim]
        # Apply DropPath regularization
        output = self.drop_path(output)
        return output

# ==========================================
# Component Combination
# ==========================================

class ComponentCombination(nn.Module):
    """Dynamically combines component outputs with learned weights."""
    def __init__(
        self, 
        input_dims: List[int], 
        hidden_dim: int = 128, 
        dropout_rate: float = 0.1, 
        activation: str = 'gelu',
        norm_type: str = 'batchnorm'
    ):
        super(ComponentCombination, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_components = len(input_dims)
        self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        
        # Choose activation function
        if activation == 'gelu':
            self.act1 = nn.GELU()
        elif activation == 'elu':
            self.act1 = nn.ELU()
        elif activation == 'leakyrelu':
            self.act1 = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Compute weights for each component
        self.fc2 = nn.Linear(hidden_dim, self.num_components)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.residual_fc = nn.Linear(sum(input_dims), sum(input_dims))
        
        # Choose normalization type
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(sum(input_dims))
        elif norm_type == 'groupnorm':
            self.norm = nn.GroupNorm(1, sum(input_dims))
        elif norm_type == 'instancenorm':
            self.norm = nn.InstanceNorm1d(sum(input_dims))
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")
        
        self.apply(initialize_weights)

    def forward(self, component_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass to combine component outputs."""
        # Check dimensions of each component output
        for i, (out, dim) in enumerate(zip(component_outputs, self.input_dims)):
            if out.shape[-1] != dim:
                raise ValueError(f"Component {i} dimension mismatch: expected {dim}, got {out.shape[-1]}")

        # Concatenate all component outputs
        concatenated = torch.cat(component_outputs, dim=-1)  # [batch, seq, sum(input_dims)]
        # Apply normalization
        x = concatenated.permute(0, 2, 1)  # [batch, sum(input_dims), seq]
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # [batch, seq, sum(input_dims)]
        # Compute residual connection
        residual = self.residual_fc(concatenated)
        # Pass through fully connected layers
        x = self.fc1(concatenated)
        x = self.act1(x)
        x = self.dropout(x)
        # Compute weights for each component
        weights = self.fc2(x)  # [batch, seq, num_components]
        weights = self.softmax(weights)
        weights = weights.split(1, dim=-1)
        # Combine weighted component outputs
        combined_output = sum(w * out for w, out in zip(weights, component_outputs))
        # Add residual connection
        combined_output += residual
        return combined_output

# ==========================================
# Tokenizers
# ==========================================

class BaseTokenizer(nn.Module):
    """Base tokenizer class."""
    def __init__(self):
        super(BaseTokenizer, self).__init__()

    def tokenize(self, data: Any) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def detokenize(self, tokens: torch.Tensor) -> str:
        raise NotImplementedError

class TextTokenizer(BaseTokenizer):
    """Tokenizer for text data using Longformer."""
    def __init__(self, encoder: LongformerTokenizer, adapt_dim: int):
        super(TextTokenizer, self).__init__()
        self.encoder = encoder
        self.vocab_size = self.encoder.vocab_size
        self.pad_token = self.encoder.pad_token_id
        self.embedding = nn.Embedding(self.vocab_size, 256)  # Уменьшено с 512
        self.adapt_dim = adapt_dim
        self.apply(initialize_weights)

    def tokenize(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize input text and convert to embeddings."""
        tokens = self.encoder.encode(text, add_special_tokens=True)
        if len(tokens) < max_length:
            tokens += [self.pad_token] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        tokens_tensor = torch.tensor(tokens)  # [seq_length]
        embeddings = self.embedding(tokens_tensor).unsqueeze(0)  # [1, seq_length, embed_dim]
        return {"tokens": tokens_tensor, "embeddings": embeddings}

    def detokenize(self, tokens: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        token_ids = tokens.cpu().numpy()
        return self.encoder.decode(token_ids, skip_special_tokens=True)

class ImageTokenizer(BaseTokenizer):
    """Tokenizer for image data using VQVAE."""
    def __init__(self, device: str = 'cpu', num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super(ImageTokenizer, self).__init__()
        self.device = device
        self.vqvae = VQVAE(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost).to(self.device)
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False

    def tokenize(self, image: Image.Image) -> torch.Tensor:
        """Tokenize image using VQVAE."""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # [1, 3, 128, 128]
        with torch.no_grad():
            vae_outputs = self.vqvae(image_tensor)
            quantized = vae_outputs["quantized"]  # [1, embedding_dim, 32, 32]
        tokens = quantized.permute(0, 2, 3, 1).contiguous().view(1, -1, quantized.shape[1])  # [1, 1024, embedding_dim]
        return tokens

    def detokenize(self, tokens: torch.Tensor) -> Image.Image:
        """Reconstruct image from tokens using VQVAE."""
        quantized = tokens.view(1, tokens.shape[-1], 32, 32)
        with torch.no_grad():
            reconstructed = self.vqvae.decoder(quantized)  # [1, 3, 128, 128]
        reconstructed = reconstructed.squeeze(0).cpu()
        reconstructed_image = transforms.ToPILImage()(reconstructed)
        return reconstructed_image

class LiquidFoundationTokenizer(nn.Module):
    """Foundation tokenizer handling both text and image modalities."""
    def __init__(self, device: str = 'cpu', adapt_dim: int = 64):  # Уменьшено с 256
        super(LiquidFoundationTokenizer, self).__init__()
        self.encoder = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.text_tokenizer = TextTokenizer(self.encoder, adapt_dim=adapt_dim)
        self.image_tokenizer = ImageTokenizer(device=device)
        self.device = device

    def tokenize(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize input data containing text and/or image."""
        tokens = {}
        if 'text' in data and data['text'] is not None:
            tokens['text'] = self.text_tokenizer.tokenize(data['text'])
        if 'image' in data and data['image'] is not None:
            tokens['image'] = self.image_tokenizer.tokenize(data['image'])
        return tokens

    def detokenize(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Detokenize tokens back to original data."""
        data = {}
        if 'text' in tokens:
            data['text'] = self.text_tokenizer.detokenize(tokens['text']['tokens'])
        if 'image' in tokens:
            data['image'] = self.image_tokenizer.detokenize(tokens['image'])
        return data

# ==========================================
# Datasets
# ==========================================

class FlickrDataset(Dataset):
    """Custom Dataset for Flickr30k with text and image."""
    def __init__(self, dataset, tokenizer: TextTokenizer, image_transform):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]
        text = sample['caption']
        image = sample['image']
        tokenized = self.tokenizer.tokenize(text)  # {'tokens': ..., 'embeddings': ...}
        image_emb = self.image_transform(image)    # [3, 128, 128]
        return {'tokens': tokenized['tokens'], 'image': image_emb}

class ChatDataset(Dataset):
    """Custom Dataset for DailyDialog with text."""
    def __init__(self, dataset, tokenizer: TextTokenizer, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.dataset[idx]
        dialog = sample['dialog']
        # Concatenate all utterances into a single string
        conversation = " ".join(dialog)
        # Tokenize the conversation
        tokenized = self.tokenizer.tokenize(conversation, max_length=self.max_length)
        return {'tokens': tokenized['tokens']}

# ==========================================
# Adaptive Configuration with Reflection Tuning
# ==========================================

class AdaptiveConfiguration(nn.Module):
    """Generates adaptive configuration weights with reflection tuning."""
    def __init__(self, adapt_dim: int, num_layers: int):
        super(AdaptiveConfiguration, self).__init__()
        self.num_layers = num_layers
        # Configuration network to generate initial weights per layer
        self.config_net = nn.Sequential(
            nn.Linear(adapt_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_layers * 4)  # 4 components per layer
        )
        self.apply(initialize_weights)
        # Reflection network to adjust weights
        self.reflection_net = nn.Sequential(
            nn.Linear(num_layers * 4, 128),
            nn.GELU(),
            nn.Linear(128, num_layers * 4)
        )
        self.apply(initialize_weights)

    def forward(self, adapt_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass to generate adaptive configuration weights."""
        # Generate initial configuration
        config = self.config_net(adapt_input)  # [batch, num_layers * 4]
        config = F.softmax(config, dim=-1)

        # Apply reflection tuning
        reflection = self.reflection_net(config)
        reflection = torch.sigmoid(reflection)

        # Adjust configuration weights
        adjusted_config = config * reflection
        adjusted_config = F.softmax(adjusted_config, dim=-1)

        # Reshape to [batch, num_layers, 4]
        adjusted_config = adjusted_config.view(-1, self.num_layers, 4)

        # Create a dictionary mapping each layer and component to its weight
        config_dict = {}
        for layer in range(self.num_layers):
            config_dict[f"layer_{layer+1}_moe_weight"] = adjusted_config[:, layer, 0]
            config_dict[f"layer_{layer+1}_token_mixer_weight"] = adjusted_config[:, layer, 1]
            config_dict[f"layer_{layer+1}_channel_mixer_weight"] = adjusted_config[:, layer, 2]
            config_dict[f"layer_{layer+1}_attention_weight"] = adjusted_config[:, layer, 3]
        return config_dict

# ==========================================
# Semantic Module
# ==========================================

class SemanticModule(nn.Module):
    """Semantics understanding module to capture complex logic and abstract concepts."""
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, adapt_dim: int, drop_prob: float = 0.1):
        super(SemanticModule, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'liquid_linear': LiquidLinear(input_dim, hidden_dim, adapt_dim),
                'attention': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=drop_prob),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(drop_prob),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(drop_prob)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(drop_prob)
            })
            for _ in range(num_layers)
        ])
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the semantic module."""
        for layer in self.layers:
            # Apply Liquid Linear layer
            x = layer['liquid_linear'](x, adapt_input)
            
            # Apply Multihead Attention
            attn_output, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + layer['dropout'](attn_output))
            
            # Apply Feed-Forward Network
            ffn_output = layer['ffn'](x)
            x = layer['norm2'](x + layer['dropout'](ffn_output))
        return x

# ==========================================
# LFModel with Gradient Checkpointing
# ==========================================

class LFModel(nn.Module):
    """Main model integrating LiquidLinear, MixtureOfExperts, attention, component combination, and semantic module."""
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        num_layers: int = 2,  # Уменьшено с 3
        hidden_dim: int = 32,  # Уменьшено с 64
        num_heads: int = 4,     # Уменьшено с 8
        semantic_hidden_dim: int = 128,  # Новая переменная
        semantic_num_heads: int = 4,
        semantic_num_layers: int = 1,
        dropout_rate: float = 0.1,
        max_drop_prob: float = 0.05,  # Уменьшено с 0.1
        layerdrop_prob: float = 0.05,  # Уменьшено с 0.1
        dropblock_block_size: int = 7,
        dropblock_prob: float = 0.05,  # Уменьшено с 0.1
        combination_activation: str = 'gelu',
        combination_norm_type: str = 'batchnorm',
        norm_type: str = 'batchnorm',
        dynamic_layer_threshold: float = 0.4  # Немного уменьшено
    ):
        super(LFModel, self).__init__()
        # Featurizer to generate adaptive input
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.featurizer.apply(initialize_weights)

        # Shared Longformer model
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        
        # DropBlock regularization
        self.dropblock = DropBlock(block_size=dropblock_block_size, drop_prob=dropblock_prob)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            drop_prob = max_drop_prob * float(i) / float(num_layers)
            layer = nn.ModuleDict({
                'token_mixer': LiquidLinear(token_dim, token_dim, adapt_dim),
                'channel_mixer': LiquidLinear(channel_dim, channel_dim, adapt_dim),
                'moe': MixtureOfExperts(
                    expert_dim, num_experts, adapt_dim, hidden_dim=hidden_dim,
                    drop_prob=drop_prob,
                    activation='gelu'
                ),
                'combiner': ComponentCombination(
                    input_dims=[token_dim, channel_dim, expert_dim, self.longformer.config.hidden_size],
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    activation=combination_activation,
                    norm_type=combination_norm_type
                ),
                'layerdrop': LayerDrop(layerdrop_prob)
            })
            self.layers.append(layer)
        
        self.dynamic_layer_threshold = dynamic_layer_threshold
        self.output_layer = nn.Linear(sum([token_dim, channel_dim, expert_dim, self.longformer.config.hidden_size]), token_dim)
        self.output_layer.apply(initialize_weights)

    def forward(self, x: torch.Tensor, config_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Forward pass through the LFModel."""
        # Generate adaptive input by averaging over sequence dimension
        adapt_input = self.featurizer(x.mean(dim=1))  # [batch, adapt_dim]
        if config_weights is None:
            # Initialize all weights to 1.0
            config_weights = {}
            for i in range(len(self.layers)):
                config_weights[f"layer_{i+1}_moe_weight"] = 1.0
                config_weights[f"layer_{i+1}_token_mixer_weight"] = 1.0
                config_weights[f"layer_{i+1}_channel_mixer_weight"] = 1.0
                config_weights[f"layer_{i+1}_attention_weight"] = 1.0
        for i, layer in enumerate(self.layers):
            layer_key_moe = f"layer_{i+1}_moe_weight"
            layer_key_token = f"layer_{i+1}_token_mixer_weight"
            layer_key_channel = f"layer_{i+1}_channel_mixer_weight"
            layer_key_attention = f"layer_{i+1}_attention_weight"
            
            layer_weight_moe = config_weights.get(layer_key_moe, 1.0)
            layer_weight_token = config_weights.get(layer_key_token, 1.0)
            layer_weight_channel = config_weights.get(layer_key_channel, 1.0)
            layer_weight_attention = config_weights.get(layer_key_attention, 1.0)
            
            # Check if all component weights are below the threshold
            if (layer_weight_moe < self.dynamic_layer_threshold and
                layer_weight_token < self.dynamic_layer_threshold and
                layer_weight_channel < self.dynamic_layer_threshold and
                layer_weight_attention < self.dynamic_layer_threshold):
                continue  # Skip this layer
            
            # Apply LayerDrop with gradient checkpointing
            x = layer['layerdrop'](x, lambda x_inner: self._process_layer(layer, x_inner, adapt_input))
        
        # Apply DropBlock regularization
        x = self.dropblock(x)
        # Final output layer
        output = self.output_layer(x)
        return output

    def _process_layer(self, layer: nn.ModuleDict, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """Processes a single layer with gradient checkpointing."""
        def custom_forward(x_inner, adapt_input_inner):
            # Apply token mixer
            token_output = layer['token_mixer'](x_inner, adapt_input_inner)
            # Apply channel mixer
            channel_output = layer['channel_mixer'](x_inner, adapt_input_inner)
            # Apply Mixture of Experts
            moe_output = layer['moe'](x_inner, adapt_input_inner)
            # Apply Longformer attention
            attention_output = self.longformer(x_inner)[0]  # [batch, seq, hidden]
            # Combine all component outputs
            component_outputs = [token_output, channel_output, moe_output, attention_output]
            combined_output = layer['combiner'](component_outputs)
            return combined_output
        return checkpoint(custom_forward, x, adapt_input)

# ==========================================
# OmniModal LLM Integrating All Components with Semantic Module
# ==========================================

class OmniModalLLM(nn.Module):
    """Omnimodal LLM handling text and image data with integrated token prediction and semantic understanding."""
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        num_layers: int = 2,  # Уменьшено с 3
        hidden_dim: int = 32,  # Уменьшено с 64
        num_heads: int = 4,     # Уменьшено с 8
        semantic_hidden_dim: int = 128,  # Новая переменная
        semantic_num_heads: int = 4,
        semantic_num_layers: int = 1,
        dropout_rate: float = 0.1,
        max_drop_prob: float = 0.05,  # Уменьшено с 0.1
        layerdrop_prob: float = 0.05,  # Уменьшено с 0.1
        dropblock_block_size: int = 7,
        dropblock_prob: float = 0.05,  # Уменьшено с 0.1
        combination_activation: str = 'gelu',
        combination_norm_type: str = 'batchnorm',
        norm_type: str = 'batchnorm',
        dynamic_layer_threshold: float = 0.4  # Немного уменьшено
    ):
        super(OmniModalLLM, self).__init__()
        # Initialize LFModel
        self.lf_model = LFModel(
            token_dim=token_dim,
            channel_dim=channel_dim,
            expert_dim=expert_dim,
            adapt_dim=adapt_dim,
            num_experts=num_experts,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            semantic_hidden_dim=semantic_hidden_dim,
            semantic_num_heads=semantic_num_heads,
            semantic_num_layers=semantic_num_layers,
            dropout_rate=dropout_rate,
            max_drop_prob=max_drop_prob,
            layerdrop_prob=layerdrop_prob,
            dropblock_block_size=dropblock_block_size,
            dropblock_prob=dropblock_prob,
            combination_activation=combination_activation,
            combination_norm_type=combination_norm_type,
            norm_type=norm_type,
            dynamic_layer_threshold=dynamic_layer_threshold
        ).to(device)
        
        # Initialize LiquidVAE
        self.liquid_vae = VQVAE(num_embeddings=512, embedding_dim=256, commitment_cost=0.25).to(device)
        
        # Initialize Adaptive Configuration
        self.adaptive_config = AdaptiveConfiguration(adapt_dim, num_layers).to(device)
        
        # Initialize Semantic Module
        self.semantic_module = SemanticModule(
            input_dim=token_dim, 
            hidden_dim=semantic_hidden_dim, 
            num_heads=semantic_num_heads, 
            num_layers=semantic_num_layers,
            adapt_dim=adapt_dim,
            drop_prob=dropblock_prob
        ).to(device)
        
        # Token predictor to generate logits over vocabulary
        self.token_predictor = nn.Linear(semantic_hidden_dim, 30522).to(device)  # Standard vocab size for Longformer
        self.token_predictor.apply(initialize_weights)

    def forward(self, text_tokens: torch.Tensor, image_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the OmniModalLLM with semantic module."""
        # Concatenate text and image tokens along sequence dimension if image is provided
        if image_embeddings is not None:
            combined_input = torch.cat([text_tokens, image_embeddings], dim=1)  # [batch, seq+img_seq]
        else:
            combined_input = text_tokens  # [batch, seq]
        
        # Generate adaptive input by averaging over sequence dimension
        adapt_input = self.lf_model.featurizer(combined_input.mean(dim=1))  # [batch, adapt_dim]
        
        # Generate adaptive configuration weights
        config = self.adaptive_config(adapt_input)
        
        # Flatten config_weights for LFModel
        config_weights = {}
        for key, value in config.items():
            config_weights[key] = value.squeeze(-1)  # Remove last dimension if necessary
        
        # Pass through LFModel
        lf_output = self.lf_model(combined_input, config_weights)  # [batch, total_seq, token_dim]
        
        # Pass through Semantic Module
        semantic_output = self.semantic_module(lf_output, adapt_input)  # [batch, total_seq, semantic_hidden_dim]
        
        # Split output into text and image parts if image is provided
        if image_embeddings is not None:
            seq_length = text_tokens.shape[1]
            text_output = semantic_output[:, :seq_length, :]  # [batch, seq, semantic_hidden_dim]
        else:
            text_output = semantic_output  # [batch, seq, semantic_hidden_dim]
        
        # Compute mean of text output for VAE
        text_mean = text_output.mean(dim=1)  # [batch, semantic_hidden_dim]
        
        # Pass through LiquidVAE
        vae_outputs = self.liquid_vae(text_mean)
        reconstructed_text = vae_outputs["reconstructed"]  # [batch, 3, 128, 128]
        
        # Generate token logits
        token_logits = self.token_predictor(text_output.mean(dim=1))  # [batch, vocab_size]
        
        # Return outputs
        return {
            "output": semantic_output,
            "token_logits": token_logits,
            "vae_reconstructed": reconstructed_text,
            "vq_loss": vae_outputs["vq_loss"],
            "perplexity": vae_outputs["perplexity"]
        }

    def save_model(self, path: str):
        """Save the model state."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load the model state."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model loaded from {path}")

# ==========================================
# Training Function
# ==========================================

def train_model(
    model,
    flickr_dataloader,
    chat_dataloader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=5,
    save_path='checkpoint.pth.tar',
    patience=3
):
    """
    Training loop handling both Flickr30k and DailyDialog datasets.

    Args:
        model (nn.Module): The multimodal model to train.
        flickr_dataloader (DataLoader): DataLoader for the Flickr30k dataset.
        chat_dataloader (DataLoader): DataLoader for the DailyDialog dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        save_path (str): Path to save the best model checkpoint.
        patience (int): Number of epochs with no improvement after which training stops.
    """
    writer = SummaryWriter()  # TensorBoard writer for monitoring
    scaler = GradScaler()      # For mixed precision training
    best_loss = float('inf')  # Initialize best loss
    epochs_no_improve = 0     # Counter for early stopping
    
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0  # Cumulative loss for the epoch
        
        # Create iterators for both dataloaders
        flickr_iter = iter(flickr_dataloader)
        chat_iter = iter(chat_dataloader)
        
        # Determine the number of batches (use the smaller size to prevent StopIteration)
        num_batches = min(len(flickr_dataloader), len(chat_dataloader))
        
        for _ in tqdm(range(num_batches), desc="Training"):
            try:
                # Get the next batch from each dataloader
                flickr_batch = next(flickr_iter)
                chat_batch = next(chat_iter)
            except StopIteration:
                # In case one dataloader is exhausted before the other
                break
            
            # Move data to the appropriate device
            tokens_flickr = flickr_batch['tokens'].to(device)      # [batch, seq]
            image_embeddings_flickr = flickr_batch['image'].to(device)    # [batch, 3, 128, 128]
            tokens_chat = chat_batch['tokens'].to(device)          # [batch, max_length]
            
            optimizer.zero_grad()  # Reset gradients
            
            with autocast():  # Enables mixed precision
                # Forward pass for Flickr30k
                outputs_flickr = model(tokens_flickr, image_embeddings_flickr)
                token_logits_flickr = outputs_flickr["token_logits"]  # [batch, vocab_size]
                
                # Prepare labels for Flickr30k
                # Shift tokens for language modeling
                labels_flickr = tokens_flickr[:, 1:].reshape(-1)  # [batch*(seq-1)]
                token_logits_flickr = token_logits_flickr[:, :-1].reshape(-1, model.token_predictor.out_features)  # [batch*(seq-1), vocab_size]
                loss_tokens_flickr = criterion(token_logits_flickr, labels_flickr)
                
                # Forward pass for DailyDialog
                outputs_chat = model(tokens_chat, image_embeddings=None)  # Chat data may not have images
                token_logits_chat = outputs_chat["token_logits"]  # [batch, vocab_size]
                
                # Prepare labels for DailyDialog
                labels_chat = tokens_chat[:, 1:].reshape(-1)  # [batch*(max_length-1)]
                token_logits_chat = token_logits_chat[:, :-1].reshape(-1, model.token_predictor.out_features)  # [batch*(max_length-1), vocab_size]
                loss_tokens_chat = criterion(token_logits_chat, labels_chat)
                
                # Total loss is the sum of both losses
                loss = loss_tokens_flickr + loss_tokens_chat
            
            # Backward pass and optimization
            scaler.scale(loss).backward()  # Scales the loss for mixed precision
            scaler.step(optimizer)         # Updates the weights
            scaler.update()                # Updates the scale for next iteration
            
            epoch_loss += loss.item()  # Accumulate loss
            
            # Clear cache to free memory
            del loss, loss_tokens_flickr, loss_tokens_chat, outputs_flickr, outputs_chat
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f"Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Save the best model checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, filename=save_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
        
        # Step the scheduler based on the average loss
        scheduler.step(avg_loss)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    
    writer.close()  # Close the TensorBoard writer

# ==========================================
# API Models
# ==========================================

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # To manage conversation sessions
    messages: List[ChatMessage]
    image: Optional[UploadFile] = None  # Optional image upload

class ChatResponse(BaseModel):
    session_id: str
    message: ChatMessage

# ==========================================
# FastAPI Setup
# ==========================================

app = FastAPI()

# Initialize Limiter for Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==========================================
# Model and Tokenizer Initialization
# ==========================================

def initialize_model_and_tokenizer(device: torch.device):
    """
    Initializes the OmniModalLLM model and the LiquidFoundationTokenizer.

    Args:
        device (torch.device): The device to load the model onto.

    Returns:
        model (OmniModalLLM): The multimodal model.
        tokenizer (LiquidFoundationTokenizer): The tokenizer for processing text and images.
    """
    token_dim = 256  # Уменьшено с 512
    channel_dim = 256  # Уменьшено с 512
    expert_dim = 128    # Уменьшено с 256
    adapt_dim = 64     # Уменьшено с 128
    num_experts = 2     # Уменьшено с 4
    num_layers = 2      # Уменьшено с 3
    hidden_dim = 32     # Уменьшено с 64
    num_heads = 4       # Уменьшено с 8
    semantic_hidden_dim = 128  # Новая переменная
    semantic_num_heads = 4      # Уменьшено с 8
    semantic_num_layers = 1     # Уменьшено с 2
    
    # Initialize the tokenizer
    tokenizer = LiquidFoundationTokenizer(device=device, adapt_dim=adapt_dim)

    # Initialize the model
    model = OmniModalLLM(
        token_dim=token_dim,
        channel_dim=channel_dim,
        expert_dim=expert_dim,
        adapt_dim=adapt_dim,
        num_experts=num_experts,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        semantic_hidden_dim=semantic_hidden_dim,
        semantic_num_heads=semantic_num_heads,
        semantic_num_layers=semantic_num_layers,
        dropout_rate=0.1,
        max_drop_prob=0.05,  # Уменьшено с 0.1
        layerdrop_prob=0.05,  # Уменьшено с 0.1
        dropblock_block_size=7,
        dropblock_prob=0.05,  # Уменьшено с 0.1
        combination_activation='gelu',
        combination_norm_type='batchnorm',
        norm_type='batchnorm',
        dynamic_layer_threshold=0.4  # Немного уменьшено
    ).to(device)

    return model, tokenizer

# Initialize the model and tokenizer
model, tokenizer = initialize_model_and_tokenizer(device=device)

# Load trained weights if available
checkpoint_path = 'final_model.pth.tar'
if os.path.exists(checkpoint_path):
    model.load_model(checkpoint_path)
else:
    print(f"No checkpoint found at '{checkpoint_path}'. Please train the model first.")

# ==========================================
# Conversation History Storage
# ==========================================

# Simple in-memory storage for session histories
conversation_history = defaultdict(list)
history_lock = threading.Lock()

# ==========================================
# Inference Function with Generation Loop
# ==========================================

def generate_response_api(
    model: OmniModalLLM,
    tokenizer: LiquidFoundationTokenizer,
    user_text: str,
    session_id: str,
    image_embeddings: Optional[torch.Tensor] = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> str:
    """
    Generates a response from the assistant based on user input and conversation history.

    Args:
        model (OmniModalLLM): The trained multimodal model.
        tokenizer (LiquidFoundationTokenizer): Tokenizer for processing text and images.
        user_text (str): The latest message from the user.
        session_id (str): Identifier for the conversation session.
        image_embeddings (Optional[torch.Tensor]): Optional image embeddings.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling.
        top_p (float): Nucleus (top-p) sampling.

    Returns:
        response_text (str): The assistant's generated response.
    """
    model.eval()  # Set model to evaluation mode
    generated_tokens = []
    with torch.no_grad():
        # Retrieve conversation history for the session
        with history_lock:
            history = conversation_history.get(session_id, []).copy()
        
        # Concatenate all user and assistant messages
        conversation = ""
        for msg in history:
            if msg.role == 'user':
                conversation += f"User: {msg.content}\n"
            elif msg.role == 'assistant':
                conversation += f"Assistant: {msg.content}\n"
        
        # Append the latest user message
        conversation += f"User: {user_text}\nAssistant:"
        
        for _ in range(max_new_tokens):
            # Tokenize the current conversation
            tokenized = tokenizer.text_tokenizer.tokenize(conversation)
            tokens = tokenized['tokens'].unsqueeze(0).to(device)  # [1, seq]
            
            # Forward pass through the model
            outputs = model(tokens, image_embeddings=image_embeddings)
            token_logits = outputs["token_logits"]  # [batch, vocab_size]
            
            # Apply temperature scaling
            token_logits = token_logits / temperature
            
            # Apply top-k and top-p filtering
            sorted_logits, sorted_indices = torch.sort(token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            token_logits.scatter_(1, indices_to_remove, float('-inf'))

            # Sample the next token
            probabilities = F.softmax(token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)  # [batch, 1]

            # Detokenize to get the token string
            token_id = next_token.squeeze(0).cpu().numpy()
            token_str = tokenizer.text_tokenizer.detokenize(next_token.squeeze(0))
            
            # Append the token to the conversation
            conversation += token_str
            generated_tokens.append(token_str)
            
            # Check for end-of-sequence token
            if tokenizer.encoder.eos_token_id and next_token.item() == tokenizer.encoder.eos_token_id:
                break

    response_text = ''.join(generated_tokens).strip()
    return response_text

# ==========================================
# API Endpoint
# ==========================================

@app.post("/chat/", response_model=ChatResponse)
@limiter.limit("20/minute")  # Limit to 20 requests per minute per IP
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    API endpoint to handle chat messages and generate responses.

    Args:
        request (ChatRequest): Incoming chat request containing messages and optional session_id.
        req (Request): The incoming HTTP request (used for rate limiting).

    Returns:
        ChatResponse: The assistant's response along with the session_id.
    """
    # Generate a new session_id if not provided
    if not request.session_id:
        session_id = str(uuid.uuid4())
        with history_lock:
            conversation_history[session_id] = []
    else:
        session_id = request.session_id
        with history_lock:
            if session_id not in conversation_history:
                conversation_history[session_id] = []
    
    # Append incoming messages to the conversation history
    with history_lock:
        conversation_history[session_id].extend(request.messages)
    
    # Extract the latest user message
    user_message = next((msg for msg in request.messages if msg.role == 'user'), None)
    if not user_message:
        return ChatResponse(
            session_id=session_id,
            message=ChatMessage(role="assistant", content="I didn't receive any user message.")
        )
    
    # Handle optional image upload
    image_embeddings = None
    if request.image:
        try:
            image = Image.open(request.image.file).convert("RGB")
            image_embeddings = tokenizer.image_tokenizer.tokenize(image).to(device)
        except Exception as e:
            return ChatResponse(
                session_id=session_id,
                message=ChatMessage(role="assistant", content=f"Failed to process the image: {str(e)}")
            )
    
    # Generate assistant's response with a generation loop
    assistant_reply = generate_response_api(
        model, 
        tokenizer, 
        user_message.content, 
        session_id=session_id,
        image_embeddings=image_embeddings,
        max_new_tokens=100,  # Adjust as needed
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    
    # Append assistant's reply to the conversation history
    with history_lock:
        conversation_history[session_id].append(
            ChatMessage(role="assistant", content=assistant_reply)
        )
    
    return ChatResponse(
        session_id=session_id,
        message=ChatMessage(role="assistant", content=assistant_reply)
    )

# ==========================================
# Entry Point
# ==========================================

def main():
    """
    Main function to handle training and serving the API.
    """
    parser = argparse.ArgumentParser(description="OmniModal LLM Training and API Server")
    parser.add_argument('--mode', type=str, choices=['train', 'serve'], required=True, help="Mode to run the script: 'train' or 'serve'")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for training")  # Уменьшено с 4
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth.tar', help="Path to save/load model checkpoints")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for optimizer")
    args = parser.parse_args()

    if args.mode == 'train':
        # ==========================================
        # Data Loading and Preparation for Training
        # ==========================================

        # Loading Flickr30k dataset
        print("Loading Flickr30k dataset...")
        flickr_dataset = load_dataset("flickr30k", split='train')
        # Filter out samples without captions or images
        flickr_dataset = flickr_dataset.filter(lambda x: len(x['caption']) > 0 and x['image'] is not None)

        # Loading DailyDialog dataset
        print("Loading DailyDialog dataset...")
        chat_dataset = load_dataset("daily_dialog", split='train')

        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Create custom Dataset instances
        flickr_custom_dataset = FlickrDataset(flickr_dataset, tokenizer.text_tokenizer, transform)
        chat_custom_dataset = ChatDataset(chat_dataset, tokenizer.text_tokenizer, max_length=512)

        # Create DataLoaders for both datasets
        flickr_dataloader = DataLoader(flickr_custom_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        chat_dataloader = DataLoader(chat_custom_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        # ==========================================
        # Optimizer, Loss, Scheduler
        # ==========================================

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        # ==========================================
        # Training
        # ==========================================

        print("Starting training...")
        train_model(
            model=model,
            flickr_dataloader=flickr_dataloader,
            chat_dataloader=chat_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            save_path=args.checkpoint,
            patience=3
        )
        print("Training completed.")

    elif args.mode == 'serve':
        # ==========================================
        # Launch the API server
        # ==========================================
        print("Launching API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()