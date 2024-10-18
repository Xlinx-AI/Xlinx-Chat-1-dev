# train.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerTokenizer, LongformerModel
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
import gradio as gr
import higher
import pandas as pd
from tqdm import tqdm
import diskcache as dc
from datasets import load_dataset
import zlib
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Any, Dict, List, Optional
import torch.utils.checkpoint as checkpoint_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, LiquidLinear):
        if hasattr(module, 'base_linear') and module.base_linear.weight is not None:
            nn.init.xavier_uniform_(module.base_linear.weight)
            if module.base_linear.bias is not None:
                nn.init.zeros_(module.base_linear.bias)
        if hasattr(module, 'adapt_linear') and module.adapt_linear.weight is not None:
            nn.init.xavier_uniform_(module.adapt_linear.weight)
            if module.adapt_linear.bias is not None:
                nn.init.zeros_(module.adapt_linear.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class DropPath(nn.Module):
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
    def __init__(self, drop_prob: float = 0.0):
        super(LayerDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, layer_fn):
        if self.drop_prob == 0.0 or not self.training:
            return layer_fn(x)
        if torch.rand(1).item() < self.drop_prob:
            return x
        return layer_fn(x)

class LiquidLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, adapt_dim: int):
        super(LiquidLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_linear = nn.Linear(in_features, out_features)
        self.adapt_linear = nn.Linear(adapt_dim, out_features * in_features)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, in_features]
        adapt_input: [batch_size, adapt_dim]
        Returns:
            [batch_size, seq_length, out_features]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x to have 3 dimensions, got {x.dim()} dimensions.")
        batch_size, seq_length, in_features = x.size()
        if in_features != self.in_features:
            raise ValueError(f"Expected input feature dimension {self.in_features}, got {in_features}.")

        # Generate adaptive weights for each example in the batch
        adapt_weight = self.adapt_linear(adapt_input)  # [batch_size, out_features * in_features]
        adapt_weight = adapt_weight.view(batch_size, self.out_features, self.in_features)  # [batch_size, out_features, in_features]

        # Expand base weights for each example in the batch
        base_weight = self.base_linear.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, out_features, in_features]

        # Final weights
        weight = base_weight + adapt_weight  # [batch_size, out_features, in_features]

        # Perform batched matrix multiplication
        # x: [batch_size, seq_length, in_features]
        # weight: [batch_size, out_features, in_features] -> [batch_size, in_features, out_features]
        weight = weight.transpose(1, 2)  # [batch_size, in_features, out_features]
        output = torch.bmm(x, weight)  # [batch_size, seq_length, out_features]

        # Add bias if exists
        if self.base_linear.bias is not None:
            output += self.base_linear.bias.unsqueeze(0).unsqueeze(1)  # [1, 1, out_features]

        return output  # [batch_size, seq_length, out_features]

class TextTokenizer(nn.Module):
    def __init__(self, encoder: LongformerTokenizer, adapt_dim: int):
        super(TextTokenizer, self).__init__()
        self.encoder = encoder
        self.vocab_size = self.encoder.vocab_size
        self.pad_token = self.encoder.pad_token_id
        self.embedding = nn.Embedding(self.vocab_size, 256)
        self.adapt_dim = adapt_dim
        self.apply(initialize_weights)

    def tokenize(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        tokens = self.encoder.encode(text, add_special_tokens=True)
        if len(tokens) < max_length:
            tokens = tokens + [self.pad_token] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
        embeddings = self.embedding(tokens_tensor).unsqueeze(0)  # Shape: [1, seq_length, embedding_dim=256]
        return {"tokens": tokens_tensor, "embeddings": embeddings}

    def detokenize(self, tokens: torch.Tensor) -> str:
        token_ids = tokens.cpu().numpy()
        return self.encoder.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Expose the embedding layer for inference
    def get_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

class LiquidFoundationTokenizer(nn.Module):
    def __init__(self, adapt_dim: int = 64):
        super(LiquidFoundationTokenizer, self).__init__()
        self.encoder = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.text_tokenizer = TextTokenizer(self.encoder, adapt_dim=adapt_dim)
        self.apply(initialize_weights)

    def tokenize(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        tokens = {}
        if 'text' in data and data['text']:
            tokens['text'] = self.text_tokenizer.tokenize(data['text'])
        return tokens

    def detokenize(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        data = {}
        if 'text' in tokens:
            data['text'] = self.text_tokenizer.detokenize(tokens['text']['tokens'])
        return data

class ChatIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer: TextTokenizer, max_length: int = 512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for sample in self.dataset:
            if isinstance(sample, dict):
                text = sample.get('text', '')  # Ensure 'text' key exists
                if len(text) > 0:
                    tokenized = self.tokenizer.tokenize(text, max_length=self.max_length)
                    yield {
                        'tokens': tokenized['tokens'],
                        'embeddings': tokenized['embeddings'].squeeze(0)  # Shape: [seq_length, embedding_dim=256]
                    }
            elif isinstance(sample, str):
                text = sample
                if len(text) > 0:
                    tokenized = self.tokenizer.tokenize(text, max_length=self.max_length)
                    yield {
                        'tokens': tokenized['tokens'],
                        'embeddings': tokenized['embeddings'].squeeze(0)  # Shape: [seq_length, embedding_dim=256]
                    }
            else:
                continue

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        flat_z = z.view(-1, self.embedding_dim)  # [batch_size * seq_length, embedding_dim]
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) +
                     torch.sum(self.embeddings.weight**2, dim=1) -
                     2 * torch.matmul(flat_z, self.embeddings.weight.t()))  # [batch_size * seq_length, num_embeddings]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [batch_size * seq_length, 1]
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encodings
        quantized = torch.matmul(encodings, self.embeddings.weight)  # [batch_size * seq_length, embedding_dim]
        quantized = quantized.view(z.shape)  # [batch_size, seq_length, embedding_dim]

        # Losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized,
            "vq_loss": loss,
            "perplexity": perplexity
        }

class VQVAE(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 256, commitment_cost: float = 0.25):
        super(VQVAE, self).__init__()
        # Encoder: Input_dim=128 (from SemanticModule's text_mean)
        self.encoder = nn.Sequential(
            LiquidLinear(128, 64, adapt_dim=64),
            nn.ReLU(),
            LiquidLinear(64, 128, adapt_dim=64),
            nn.ReLU(),
            LiquidLinear(128, embedding_dim, adapt_dim=64)
        )
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # Decoder: Output_dim=128
        self.decoder = nn.Sequential(
            LiquidLinear(embedding_dim, 128, adapt_dim=64),
            nn.ReLU(),
            LiquidLinear(128, 64, adapt_dim=64),
            nn.ReLU(),
            LiquidLinear(64, 128, adapt_dim=64),
            nn.Sigmoid()
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor = None):
        if adapt_input is None:
            adapt_input = torch.zeros(x.size(0), 64).to(x.device)
        # Ensure x is 2D [batch_size, 128]
        if x.dim() != 2:
            raise ValueError(f"Expected x to have 2 dimensions, got {x.dim()} dimensions.")
        z_e = self.encoder(x, adapt_input)  # [batch_size, embedding_dim=256]
        vq_outputs = self.vq_layer(z_e)  # Quantized tensor and losses
        z_q = vq_outputs["quantized"]  # [batch_size, 256]
        x_recon = self.decoder(z_q, adapt_input)  # [batch_size, 128]
        return {
            "quantized": z_q,  # [batch_size, 256]
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "reconstructed": x_recon  # [batch_size, 128]
        }

class KolmogorovArnoldExpert(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, activation: str = 'gelu'):
        super(KolmogorovArnoldExpert, self).__init__()
        act_fn = {'gelu': nn.GELU(), 'elu': nn.ELU(), 'leakyrelu': nn.LeakyReLU()}[activation]
        self.phi_functions = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), act_fn) for _ in range(input_dim)
        ])
        self.psi_function = nn.Sequential(
            nn.Linear(input_dim * hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim)
        )
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, input_dim]
        Returns:
            [batch_size, seq_length, output_dim]
        """
        phi_outputs = [
            phi(x[:, :, i].unsqueeze(2)) for i, phi in enumerate(self.phi_functions)
        ]  # List of [batch_size, seq_length, hidden_dim]
        concatenated = torch.cat(phi_outputs, dim=-1)  # [batch_size, seq_length, input_dim * hidden_dim]
        return self.psi_function(concatenated)  # [batch_size, seq_length, output_dim]

class MixtureOfExperts(nn.Module):
    def __init__(self, expert_dim: int, num_experts: int, adapt_dim: int, hidden_dim: int = 64, drop_prob: float = 0.0, activation: str = 'gelu'):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([
            LiquidLinear(expert_dim, expert_dim, adapt_dim) for _ in range(num_experts)
        ])
        self.ka_expert = KolmogorovArnoldExpert(expert_dim, expert_dim, hidden_dim, activation)
        self.gating = nn.Linear(adapt_dim, num_experts + 1)
        self.drop_path = DropPath(drop_prob)
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, expert_dim]
        adapt_input: [batch_size, adapt_dim]
        Returns:
            [batch_size, seq_length, expert_dim]
        """
        gate_scores = F.softmax(self.gating(adapt_input), dim=-1)  # [batch_size, num_experts +1]
        gate_scores = gate_scores.unsqueeze(1)  # [batch_size, 1, num_experts +1]
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x, adapt_input)  # [batch_size, seq_length, expert_dim]
            gate = gate_scores[:, :, i].unsqueeze(-1)  # [batch_size, 1, 1]
            expert_outputs.append(gate * expert_out)  # [batch_size, seq_length, expert_dim]
        # Add KolmogorovArnoldExpert
        ka_out = gate_scores[:, :, -1].unsqueeze(-1) * self.ka_expert(x)  # [batch_size, seq_length, expert_dim]
        expert_outputs.append(ka_out)
        # Sum all expert outputs
        output = torch.stack(expert_outputs, dim=-1).sum(dim=-1)  # [batch_size, seq_length, expert_dim]
        return self.drop_path(output)  # [batch_size, seq_length, expert_dim]

class ComponentCombination(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int = 128, dropout_rate: float = 0.1, activation: str = 'gelu', norm_type: str = 'batchnorm'):
        super(ComponentCombination, self).__init__()
        self.fc1 = nn.Linear(sum(input_dims), hidden_dim)
        self.act1 = {'gelu': nn.GELU(), 'elu': nn.ELU(), 'leakyrelu': nn.LeakyReLU()}[activation]
        self.fc2 = nn.Linear(hidden_dim, len(input_dims))
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.residual_fc = nn.Linear(sum(input_dims), sum(input_dims))
        self.norm = {
            'batchnorm': nn.BatchNorm1d(sum(input_dims)),
            'groupnorm': nn.GroupNorm(1, sum(input_dims)),
            'instancenorm': nn.InstanceNorm1d(sum(input_dims))
        }[norm_type]
        self.apply(initialize_weights)

    def forward(self, component_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        component_outputs: List of [batch_size, seq_length, feature_dim_i]
        Returns:
            [batch_size, seq_length, sum(input_dims)]
        """
        concatenated = torch.cat(component_outputs, dim=-1)  # [batch_size, seq_length, sum(input_dims)]
        # Normalize
        x = self.norm(concatenated.permute(0, 2, 1))  # [batch_size, sum(input_dims), seq_length]
        x = x.permute(0, 2, 1)  # [batch_size, seq_length, sum(input_dims)]
        # Residual connection
        residual = self.residual_fc(concatenated)  # [batch_size, seq_length, sum(input_dims)]
        # Apply FC1 and activation
        x = self.fc1(concatenated)  # [batch_size, seq_length, hidden_dim]
        x = self.act1(x)
        x = self.dropout(x)
        # Get weights for components
        weights = self.fc2(x)  # [batch_size, seq_length, len(input_dims)]
        weights = self.softmax(weights)  # [batch_size, seq_length, len(input_dims)]
        # Apply weights to components
        combined_output = torch.zeros_like(concatenated)
        for i, out in enumerate(component_outputs):
            combined_output += weights[:, :, i].unsqueeze(-1) * out  # [batch_size, seq_length, feature_dim_i]
        # Add residual
        combined_output += residual  # [batch_size, seq_length, sum(input_dims)]
        return combined_output  # [batch_size, seq_length, sum(input_dims)]

class AdaptiveConfiguration(nn.Module):
    def __init__(self, adapt_dim: int, num_layers: int):
        super(AdaptiveConfiguration, self).__init__()
        self.num_layers = num_layers
        self.config_net = nn.Sequential(
            nn.Linear(adapt_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_layers * 4)
        )
        self.reflection_net = nn.Sequential(
            nn.Linear(num_layers * 4, 128),
            nn.GELU(),
            nn.Linear(128, num_layers * 4)
        )
        self.apply(initialize_weights)

    def forward(self, adapt_input: torch.Tensor) -> Dict[str, float]:
        """
        adapt_input: [batch_size, adapt_dim]
        Returns:
            Dict[str, float] with weights for each layer component
        """
        config = self.config_net(adapt_input)  # Shape: [batch_size, num_layers * 4]
        config = F.softmax(config, dim=-1)
        reflection = self.reflection_net(config)  # Shape: [batch_size, num_layers * 4]
        reflection = torch.sigmoid(reflection)
        adjusted_config = config * reflection
        adjusted_config = F.softmax(adjusted_config, dim=-1)
        adjusted_config = adjusted_config.view(-1, self.num_layers, 4)  # [batch_size, num_layers, 4]
        config_dict = {}
        for layer in range(self.num_layers):
            # Average over the batch to obtain scalar weights
            config_dict[f"layer_{layer+1}_moe_weight"] = adjusted_config[:, layer, 0].mean().item()
            config_dict[f"layer_{layer+1}_token_mixer_weight"] = adjusted_config[:, layer, 1].mean().item()
            config_dict[f"layer_{layer+1}_channel_mixer_weight"] = adjusted_config[:, layer, 2].mean().item()
            config_dict[f"layer_{layer+1}_attention_weight"] = adjusted_config[:, layer, 3].mean().item()
        return config_dict

class SemanticModule(nn.Module):
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
        """
        x: [batch_size, seq_length, input_dim]
        adapt_input: [batch_size, adapt_dim]
        Returns:
            [batch_size, seq_length, hidden_dim]
        """
        for layer in self.layers:
            # Apply LiquidLinear
            residual = layer['liquid_linear'](x, adapt_input)  # [batch_size, seq_length, hidden_dim]
            # Prepare for MultiheadAttention: [seq_length, batch_size, hidden_dim]
            attn_input = residual.transpose(0, 1)  # [seq_length, batch_size, hidden_dim]
            attn_output, _ = layer['attention'](attn_input, attn_input, attn_input)  # [seq_length, batch_size, hidden_dim]
            # Convert back to [batch_size, seq_length, hidden_dim]
            attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_length, hidden_dim]
            attn_output = layer['dropout'](attn_output)
            # Normalize and add residual
            x = layer['norm1'](x + attn_output)  # [batch_size, seq_length, hidden_dim]
            # Apply Feed-Forward Network
            ffn_output = layer['ffn'](x)  # [batch_size, seq_length, hidden_dim]
            ffn_output = layer['dropout'](ffn_output)
            # Normalize and add residual
            x = layer['norm2'](x + ffn_output)  # [batch_size, seq_length, hidden_dim]
        return x  # [batch_size, seq_length, hidden_dim]

class LFModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        channel_dim: int,
        expert_dim: int,
        adapt_dim: int,
        num_experts: int,
        num_layers: int = 2,
        hidden_dim: int = 32,
        num_heads: int = 4,
        semantic_hidden_dim: int = 128,
        semantic_num_heads: int = 4,
        semantic_num_layers: int = 1,
        dropout_rate: float = 0.1,
        max_drop_prob: float = 0.05,
        layerdrop_prob: float = 0.05,
        dropblock_block_size: int = 7,
        dropblock_prob: float = 0.05,
        combination_activation: str = 'gelu',
        combination_norm_type: str = 'batchnorm',
        norm_type: str = 'batchnorm',
        dynamic_layer_threshold: float = 0.4
    ):
        super(LFModel, self).__init__()
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        self.featurizer.apply(initialize_weights)
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096').to(device)
        self.longformer.eval()
        for param in self.longformer.parameters():
            param.requires_grad = False
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
        # Projection layer: [combined_features=5120] -> [256]
        self.output_layer = nn.Linear(5120, 256)
        self.output_layer.apply(initialize_weights)

    def forward(self, x: torch.Tensor, config_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_length, token_dim=256]
        config_weights: Dict[str, float]
        Returns:
            [batch_size, seq_length, token_dim=256]
        """
        adapt_input = self.featurizer(x.mean(dim=1))  # [batch_size, adapt_dim]
        if config_weights is None:
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
            if (layer_weight_moe < self.dynamic_layer_threshold and
                layer_weight_token < self.dynamic_layer_threshold and
                layer_weight_channel < self.dynamic_layer_threshold and
                layer_weight_attention < self.dynamic_layer_threshold):
                continue
            # Apply layerdrop with processing function
            x = layer['layerdrop'](x, lambda x_inner: self._process_layer(layer, x_inner, adapt_input))
        x = self.dropblock(x)
        # Apply output_layer to all tokens simultaneously
        output = self.output_layer(x)  # [batch_size, seq_length, 256]
        return output

    def _process_layer(self, layer: nn.ModuleDict, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_length, token_dim=256]
        adapt_input: [batch_size, adapt_dim]
        Returns:
            [batch_size, seq_length, combined_features=5120]
        """
        # Apply LiquidLinear layers
        token_output = layer['token_mixer'](x, adapt_input)  # [batch_size, seq_length, 256]
        channel_output = layer['channel_mixer'](x, adapt_input)  # [batch_size, seq_length, 256]
        moe_output = layer['moe'](x, adapt_input)  # [batch_size, seq_length, 128]
        # Get attention_output from Longformer
        longformer_output = self.longformer(x)[0]  # [batch_size, seq_length, hidden_size=4096]
        attention_output = longformer_output.mean(dim=1)  # [batch_size, hidden_size=4096]
        attention_output = attention_output.unsqueeze(1).repeat(1, x.size(1), 1)  # [batch_size, seq_length, 4096]
        # Combine all components
        component_outputs = [token_output, channel_output, moe_output, attention_output]  # [256, 256, 128, 4096]
        combined_output = layer['combiner'](component_outputs)  # [batch_size, seq_length, 5120]
        return combined_output

class XlinxChatModel(nn.Module):
    def __init__(
        self,
        token_dim: int = 256,
        channel_dim: int = 256,
        expert_dim: int = 128,
        adapt_dim: int = 64,
        num_experts: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 32,
        num_heads: int = 4,
        semantic_hidden_dim: int = 128,
        semantic_num_heads: int = 4,
        semantic_num_layers: int = 1,
        dropout_rate: float = 0.1,
        max_drop_prob: float = 0.05,
        layerdrop_prob: float = 0.05,
        dropblock_block_size: int = 7,
        dropblock_prob: float = 0.05,
        combination_activation: str = 'gelu',
        combination_norm_type: str = 'batchnorm',
        norm_type: str = 'batchnorm',
        dynamic_layer_threshold: float = 0.4
    ):
        super(XlinxChatModel, self).__init__()
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
        self.liquid_vae = VQVAE(num_embeddings=512, embedding_dim=256, commitment_cost=0.25).to(device)
        self.liquid_vae.eval()
        for param in self.liquid_vae.parameters():
            param.requires_grad = False
        self.adaptive_config = AdaptiveConfiguration(adapt_dim, num_layers).to(device)
        self.semantic_module = SemanticModule(
            input_dim=256,  # token_dim
            hidden_dim=semantic_hidden_dim, 
            num_heads=semantic_num_heads, 
            num_layers=semantic_num_layers,
            adapt_dim=adapt_dim,
            drop_prob=dropblock_prob
        ).to(device)
        self.token_predictor = nn.Linear(semantic_hidden_dim, 30522).to(device)  # Assuming vocabulary size 30522
        self.token_predictor.apply(initialize_weights)
        self.cache = dc.Cache('./cache')

    def forward(self, text_tokens: torch.Tensor, image_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        text_tokens: [batch_size, seq_length, embedding_dim=256]
        image_embeddings: [batch_size, image_dim] (Optional)
        Returns:
            {
                "output": [batch_size, seq_length, hidden_dim=128],
                "token_logits": [batch_size, seq_length, 30522],
                "vae_reconstructed": [batch_size, 128],
                "vq_loss": scalar,
                "perplexity": scalar
            }
        """
        # Ensure input dimensions
        if text_tokens.dim() != 3 or text_tokens.size(2) != 256:
            raise ValueError(f"Expected text_tokens to have shape [batch_size, seq_length, 256], got {text_tokens.shape}")

        combined_input = text_tokens  # [batch_size, seq_length, 256]
        adapt_input = self.lf_model.featurizer(combined_input.mean(dim=1))  # [batch_size, adapt_dim]

        # Obtain adaptive configuration
        config = self.adaptive_config(adapt_input)  # Dict[str, float]
        config_weights = {key: value for key, value in config.items()}  # Scalars

        # Pass through LFModel
        lf_output = self.lf_model(combined_input, config_weights)  # [batch_size, seq_length, 5120]

        # Pass through SemanticModule
        semantic_output = self.semantic_module(lf_output, adapt_input)  # [batch_size, seq_length, 128]
        text_output = semantic_output  # [batch_size, seq_length, 128]

        # Mean pooling for VQVAE
        text_mean = text_output.mean(dim=1)  # [batch_size, 128]

        # Pass through VQVAE
        vae_outputs = self.liquid_vae(text_mean)  # Dict with reconstructed, vq_loss, perplexity
        reconstructed_text = vae_outputs["reconstructed"]  # [batch_size, 128]

        # Token prediction
        token_logits = self.token_predictor(semantic_output)  # [batch_size, seq_length, 30522]

        return {
            "output": semantic_output,  # [batch_size, seq_length, 128]
            "token_logits": token_logits,  # [batch_size, seq_length, 30522]
            "vae_reconstructed": reconstructed_text,  # [batch_size, 128]
            "vq_loss": vae_outputs["vq_loss"],
            "perplexity": vae_outputs["perplexity"]
        }

    def compress_tensor(self, tensor: torch.Tensor) -> int:
        tensor_np = tensor.cpu().numpy()
        compressed = zlib.compress(tensor_np.tobytes())
        key = hash(tensor.data_ptr())
        self.cache.set(key, compressed)
        self.cache.set(f"{key}_shape", tensor.shape)
        return key

    def decompress_tensor(self, key: int) -> torch.Tensor:
        tensor_bytes = self.cache.get(key)
        shape = self.cache.get(f"{key}_shape")
        if tensor_bytes is None or shape is None:
            raise ValueError("Data not found in cache.")
        tensor_np = zlib.decompress(tensor_bytes)
        tensor_np = np.frombuffer(tensor_np, dtype=np.float32)
        tensor = torch.from_numpy(tensor_np).view(*shape).to(device)
        return tensor

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)

class MetaLearner:
    def __init__(self, model, inner_lr=1e-2, meta_lr=1e-3, device='cuda'):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.device = device
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)

    def meta_train_step(self, task_support, task_query, criterion):
        support_inputs, support_targets = task_support
        query_inputs, query_targets = task_query
        support_inputs = support_inputs.to(self.device)
        support_targets = support_targets.to(self.device)
        query_inputs = query_inputs.to(self.device)
        query_targets = query_targets.to(self.device)
        self.meta_optimizer.zero_grad()
        with higher.innerloop_ctx(self.model, torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)) as (fmodel, diffopt):
            support_outputs = fmodel(support_inputs, image_embeddings=None)
            support_loss = criterion(support_outputs["token_logits"], support_targets)
            diffopt.step(support_loss)
            query_outputs = fmodel(query_inputs, image_embeddings=None)
            query_loss = criterion(query_outputs["token_logits"], query_targets)
        query_loss.backward()
        self.meta_optimizer.step()
        return support_loss.item(), query_loss.item()

def train_model_meta(
    model,
    flickr_dataloader,
    chat_dataloader,
    meta_learner,
    criterion,
    scheduler,
    device,
    num_epochs=5,
    save_path='checkpoint.pth.tar',
    patience=3,
    accumulation_steps=4
):
    writer = SummaryWriter()
    best_loss = float('inf')
    epochs_no_improve = 0
    scaler = GradScaler() if device.type == 'cuda' else None
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        flickr_iter = iter(flickr_dataloader)
        chat_iter = iter(chat_dataloader)
        steps = 0
        for step in tqdm(range(10), desc="Meta-Training"):
            try:
                flickr_batch = next(flickr_iter)
                chat_batch = next(chat_iter)
            except StopIteration:
                break
            support_batch = flickr_batch
            query_batch = chat_batch
            # Используем embeddings вместо токенов
            support_inputs = support_batch['embeddings'].unsqueeze(1).to(device)  # [batch_size, 1, 256]
            support_targets = support_batch['tokens'][:, 1:].contiguous().to(device)  # [batch_size, seq_length -1]
            query_inputs = query_batch['embeddings'].unsqueeze(1).to(device)  # [batch_size, 1, 256]
            query_targets = query_batch['tokens'][:, 1:].contiguous().to(device)  # [batch_size, seq_length -1]
            if scaler:
                with autocast():
                    support_outputs = model(support_inputs, image_embeddings=None)
                    support_loss = criterion(support_outputs["token_logits"], support_targets)
                    support_loss = support_loss / accumulation_steps
                scaler.scale(support_loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(meta_learner.meta_optimizer)
                    scaler.update()
                    meta_learner.meta_optimizer.zero_grad()
            else:
                support_outputs = model(support_inputs, image_embeddings=None)
                support_loss = criterion(support_outputs["token_logits"], support_targets)
                support_loss = support_loss / accumulation_steps
                support_loss.backward()
                if (step + 1) % accumulation_steps == 0:
                    meta_learner.meta_optimizer.step()
                    meta_learner.meta_optimizer.zero_grad()
            support_loss_item, query_loss_item = meta_learner.meta_train_step(
                task_support=(support_inputs, support_targets),
                task_query=(query_inputs, query_targets),
                criterion=criterion
            )
            epoch_loss += query_loss_item
            steps += 1
            if steps >= len(flickr_dataloader):
                break
        avg_loss = epoch_loss / steps if steps > 0 else 0
        writer.add_scalar('Loss/meta_train', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': meta_learner.meta_optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        scheduler.step(avg_loss)
        writer.add_scalar('Learning Rate', meta_learner.meta_optimizer.param_groups[0]['lr'], epoch)
    writer.close()

def generate_response_api(
    model: 'XlinxChatModel',
    tokenizer: 'LiquidFoundationTokenizer',
    user_text: str,
    session_id: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> str:
    model.eval()
    generated_tokens = []
    with torch.no_grad():
        conversation_history = "User: " + user_text + "\nAssistant:"
        tokenized = tokenizer.text_tokenizer.tokenize(conversation_history)
        embeddings = tokenized['embeddings'].to(device)  # [1, seq_length, 256]
        tokens = embeddings  # [1, seq_length, 256]
        for _ in range(max_new_tokens):
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(tokens, image_embeddings=None)
                token_logits = outputs["token_logits"]  # [1, seq_length, 30522]
                # Берём последний токен для генерации следующего
                last_token_logits = token_logits[:, -1, :]  # [1, 30522]
                last_token_logits = last_token_logits / temperature
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_token_logits.scatter_(1, indices_to_remove, float('-inf'))
                probabilities = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)  # [1, 1]
                token_str = tokenizer.text_tokenizer.detokenize(next_token.squeeze(0))
                generated_tokens.append(token_str)
                conversation_history += token_str
                if tokenizer.encoder.eos_token_id and next_token.item() == tokenizer.encoder.eos_token_id:
                    break
                # Получаем эмбеддинг для нового токена
                new_embedding = tokenizer.text_tokenizer.get_embedding(next_token).unsqueeze(0)  # [1, 1, 256]
                tokens = torch.cat([tokens, new_embedding.to(device)], dim=1)  # [1, seq_length +1, 256]
    response_text = ''.join(generated_tokens).strip()
    return response_text

def generate_response_gradio(user_text):
    assistant_reply = generate_response_api(
        model, 
        tokenizer, 
        user_text, 
        session_id="gradio_session",
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    return assistant_reply

def initialize_model_and_tokenizer(device: torch.device):
    token_dim = 256
    channel_dim = 256
    expert_dim = 128
    adapt_dim = 64
    num_experts = 4
    num_layers = 2
    hidden_dim = 32
    num_heads = 4
    semantic_hidden_dim = 128
    semantic_num_heads = 4
    semantic_num_layers = 1
    tokenizer = LiquidFoundationTokenizer(adapt_dim=adapt_dim).to(device)
    model = XlinxChatModel(
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
        max_drop_prob=0.05,
        layerdrop_prob=0.05,
        dropblock_block_size=7,
        dropblock_prob=0.05,
        combination_activation='gelu',
        combination_norm_type='batchnorm',
        norm_type='batchnorm',
        dynamic_layer_threshold=0.4
    ).to(device)
    return model, tokenizer

def main():
    
    parser = argparse.ArgumentParser(description="XlinxChatModel with Gradio and Meta-Learning")
    parser.add_argument('--mode', type=str, choices=['train', 'serve'], required=True, help="Mode: 'train' or 'serve'")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size")
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth.tar', help="Checkpoint path")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--meta', action='store_true', help="Enable meta-learning")
    parser.add_argument('--accumulation_steps', type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    global model, tokenizer
    model, tokenizer = initialize_model_and_tokenizer(device=device)

    if args.mode == 'train':
        try:
            chat_dataset = load_dataset("Thewillonline/gpt4", split='train')
        except Exception as e:
            print(f"Error loading Chat Dataset: {e}")
            return

        chat_custom_dataset = ChatIterableDataset(chat_dataset, tokenizer.text_tokenizer, max_length=512)
        chat_dataloader = DataLoader(chat_custom_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

        if args.meta:
            meta_learner = MetaLearner(
                model=model,
                inner_lr=1e-2,
                meta_lr=args.learning_rate,
                device=device
            )
            train_model_meta(
                model=model,
                flickr_dataloader=chat_dataloader,  # Using chat_dataloader as placeholder
                chat_dataloader=chat_dataloader,
                meta_learner=meta_learner,
                criterion=criterion,
                scheduler=scheduler,
                device=device,
                num_epochs=args.epochs,
                save_path=args.checkpoint,
                patience=3,
                accumulation_steps=args.accumulation_steps
            )
        else:
            writer = SummaryWriter()
            best_loss = float('inf')
            epochs_no_improve = 0
            scaler = GradScaler() if device.type == 'cuda' else None
            model.train()
            for epoch in range(args.epochs):
                epoch_loss = 0.0
                chat_iter = iter(chat_dataloader)
                steps = 0
                for step in tqdm(range(10), desc="Training"):
                    try:
                        chat_batch = next(chat_iter)
                    except StopIteration:
                        break
                    support_batch = chat_batch
                    # Используем embeddings вместо токенов
                    support_inputs = support_batch['embeddings'].unsqueeze(1).to(device)  # [batch_size, 1, 256]
                    support_targets = support_batch['tokens'][:, 1:].contiguous().to(device)  # [batch_size, seq_length -1]
                    if scaler:
                        with autocast():
                            outputs = model(support_inputs, image_embeddings=None)
                            loss = criterion(outputs["token_logits"][:, -1, :], support_targets)  # Берём только последний токен
                            loss = loss / args.accumulation_steps
                        scaler.scale(loss).backward()
                        if (step + 1) % args.accumulation_steps == 0:
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = model(support_inputs, image_embeddings=None)
                        loss = criterion(outputs["token_logits"][:, -1, :], support_targets)  # Берём только последний токен
                        loss = loss / args.accumulation_steps
                        loss.backward()
                        if (step + 1) % args.accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    epoch_loss += loss.item() * args.accumulation_steps
                    steps += 1
                    if steps >= len(chat_dataloader):
                        break
                avg_loss = epoch_loss / steps if steps > 0 else 0
                writer.add_scalar('Loss/train', avg_loss, epoch)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    epochs_no_improve = 0
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, args.checkpoint)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= 3:
                        print("Early stopping triggered.")
                        break
                scheduler.step(avg_loss)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            writer.close()
    elif args.mode == 'serve':
        if os.path.exists(args.checkpoint):
            model.load_model(args.checkpoint)
            print(f"Loaded model checkpoint from '{args.checkpoint}'.")
        else:
            print(f"No checkpoint found at '{args.checkpoint}'. Please train the model first.")
            return
        iface = gr.Interface(
            fn=generate_response_gradio,
            inputs=[
                gr.inputs.Textbox(lines=2, placeholder="Введите ваше сообщение здесь...")
            ],
            outputs="text",
            title="XlinxChatModel Chatbot",
            description="Чат-бот с возможностями AGI, продвинутым рассуждением и саморегуляцией.",
            examples=[
                ["Привет, как дела?"],
                ["Расскажи мне историю об искусственном интеллекте."]
            ],
            live=False
        )
        iface.launch()

if __name__ == "__main__":
    
    model = XlinxChatModel(
        token_dim=256,
        channel_dim=256,
        expert_dim=128,
        adapt_dim=64,
        num_experts=4,
        num_layers=2,
        hidden_dim=32,
        num_heads=4,
        semantic_hidden_dim=128,
        semantic_num_heads=4,
        semantic_num_layers=1,
        dropout_rate=0.1,
        max_drop_prob=0.05,
        layerdrop_prob=0.05,
        dropblock_block_size=7,
        dropblock_prob=0.05,
        combination_activation='gelu',
        combination_norm_type='batchnorm',
        norm_type='batchnorm',
        dynamic_layer_threshold=0.4
    ).to(device)
    
    # Создание фиктивных входных данных для проверки модели
    batch_size = 2
    seq_length = 10
    text_tokens = torch.randn(batch_size, seq_length, 256).to(device)  # [batch_size, seq_length, 256]
    
    # Прямой проход
    outputs = model(text_tokens)
    
    # Печать размерностей выходов
    print("Output Shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    main()
