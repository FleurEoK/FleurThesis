# Import necessary modules
from functools import partial  # Used to partially initialize functions with fixed arguments

import torch  # PyTorch library for deep learning
import torch.nn as nn  # PyTorch's neural network module

import timm.models.vision_transformer  # Vision Transformer models from the TIMM library



# Define a custom Vision Transformer class
class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with support for global average pooling.
    Extends the VisionTransformer from TIMM.
    """
    def __init__(self, global_pool=False, **kwargs):
        """
        Initialize the Vision Transformer.

        Args:
            global_pool (bool): If True, enable global average pooling.
            **kwargs: Additional arguments for the base VisionTransformer class.
        """
        super(VisionTransformer, self).__init__(**kwargs)  # Initialize the base class

        self.global_pool = global_pool  # Store whether to use global pooling
        if self.global_pool:
            # Extract the normalization layer and embedding dimension from kwargs
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']

            # Add a custom fully connected normalization layer
            self.fc_norm = norm_layer(embed_dim)

            # Remove the original normalization layer from the base class
            del self.norm

    def forward_features(self, x):
        """
        Forward pass for feature extraction.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Extracted features.
        """
        B = x.shape[0]  # Batch size

        # Embed input patches into the transformer input format
        x = self.patch_embed(x)

        # Add class token to each batch and concatenate it with the patch embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings to the input
        x = x + self.pos_embed
        x = self.pos_drop(x)  # Apply dropout to the positional embeddings

        # Pass the input through the transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Perform global average pooling if enabled
        if self.global_pool:
            # Pool over all tokens except the class token
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)  # Apply normalization to the pooled output
        else:
            # Apply the original normalization and extract the class token output
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome  # Return the final output features

    def forward_head(self, x):
        """
        Pass the features through the classification head.

        Args:
            x (torch.Tensor): Extracted features.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.head(x)  # Apply the classification head

# Functions to instantiate Vision Transformer models with varying configurations
def vit_small_patch16(**kwargs):
    """
    Create a small Vision Transformer with 16x16 patch size.
    """
    model = VisionTransformer(
        patch_size=16,          # Size of input patches (16x16 pixels)
        embed_dim=384,          # Embedding dimension
        depth=12,               # Number of transformer blocks
        num_heads=6,            # Number of attention heads
        mlp_ratio=4,            # Ratio for the hidden layer in MLP
        qkv_bias=True,          # Use bias in QKV projections
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Layer normalization with small epsilon
        **kwargs                # Additional arguments
    )
    return model

def vit_base_patch16(**kwargs):
    """
    Create a base Vision Transformer with 16x16 patch size.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,          # Larger embedding dimension than the small variant
        depth=12,               # Same depth as the small variant
        num_heads=12,           # More attention heads for better representation
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_large_patch16(**kwargs):
    """
    Create a large Vision Transformer with 16x16 patch size.
    """
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,         # Even larger embedding dimension
        depth=24,               # Twice the depth of the base model
        num_heads=16,           # More attention heads for capturing richer information
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_huge_patch14(**kwargs):
    """
    Create a huge Vision Transformer with 14x14 patch size.
    """
    model = VisionTransformer(
        patch_size=14,          # Smaller patch size for finer granularity
        embed_dim=1280,         # Massive embedding dimension
        depth=32,               # Deepest transformer with 32 layers
        num_heads=16,           # Same number of attention heads as the large model
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
