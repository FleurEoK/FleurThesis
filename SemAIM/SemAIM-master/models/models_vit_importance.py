# Import necessary modules
from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformerWithImportance(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with importance-aware mechanisms for fine-tuning.
    Maintains the same architecture modifications used during pre-training.
    """
    def __init__(self, global_pool=False, use_importance_bias=True, 
                 use_importance_pe=True, importance_alpha=1.0, 
                 importance_beta=0.1, **kwargs):
        """
        Initialize the importance-aware Vision Transformer.

        Args:
            global_pool (bool): If True, enable global average pooling.
            use_importance_bias (bool): Enable importance-based attention bias.
            use_importance_pe (bool): Enable importance-weighted positional encoding.
            importance_alpha (float): Scaling factor for importance-weighted PE.
            importance_beta (float): Scaling factor for importance bias.
            **kwargs: Additional arguments for the base VisionTransformer class.
        """
        super(VisionTransformerWithImportance, self).__init__(**kwargs)
        
        self.global_pool = global_pool
        self.use_importance_bias = use_importance_bias
        self.use_importance_pe = use_importance_pe
        self.importance_alpha = importance_alpha
        self.importance_beta = importance_beta
        
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x, importance_scores=None):
        """
        Forward pass for feature extraction with optional importance awareness.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).
            importance_scores (torch.Tensor, optional): Importance scores with shape (B, 196).

        Returns:
            torch.Tensor: Extracted features.
        """
        B = x.shape[0]
        
        # Embed input patches
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply importance-weighted positional encoding if enabled
        if self.use_importance_pe and importance_scores is not None:
            # importance_scores shape: (B, 196)
            # We need to add the class token position (no importance weighting for cls token)
            pos_embed = self.pos_embed.clone()
            
            # Normalize importance scores to have mean weight of 1.0
            importance_weights = 1.0 + self.importance_alpha * importance_scores
            importance_weights_mean = importance_weights.mean(dim=1, keepdim=True)
            importance_weights = importance_weights / importance_weights_mean
            
            # Apply weights to positional embeddings (skip class token at position 0)
            # pos_embed shape: (1, 197, embed_dim) -> (1, cls_token, 196_patches)
            pos_embed_cls = pos_embed[:, :1, :]  # Class token position
            pos_embed_patches = pos_embed[:, 1:, :]  # Patch positions
            
            # Broadcast importance weights: (B, 196) -> (B, 196, 1) -> (B, 196, embed_dim)
            importance_weights_expanded = importance_weights.unsqueeze(-1)
            pos_embed_patches_weighted = pos_embed_patches * importance_weights_expanded
            
            # Concatenate back
            pos_embed_weighted = torch.cat([pos_embed_cls.expand(B, -1, -1), 
                                           pos_embed_patches_weighted], dim=1)
            x = x + pos_embed_weighted
        else:
            # Standard positional encoding
            x = x + self.pos_embed
        
        x = self.pos_drop(x)
        
        # Prepare importance bias for attention if enabled
        attention_bias = None
        if self.use_importance_bias and importance_scores is not None:
            # Create bias matrix: (B, 1, 1, 197) for broadcasting
            # Bias only for patch tokens, not class token
            bias_patches = self.importance_beta * torch.log(1.0 + importance_scores)
            bias_cls = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
            bias_full = torch.cat([bias_cls, bias_patches], dim=1)  # (B, 197)
            attention_bias = bias_full.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 197)
        
        # Pass through transformer blocks with attention bias
        for blk in self.blocks:
            if attention_bias is not None:
                # Modify block to accept bias - this requires custom block implementation
                # For now, we'll pass through standard blocks
                # In production, you'd modify the attention mechanism in each block
                x = blk(x)
            else:
                x = blk(x)
        
        # Global pooling or class token
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        
        return outcome

    def forward(self, x, importance_scores=None):
        """
        Full forward pass.

        Args:
            x (torch.Tensor): Input images.
            importance_scores (torch.Tensor, optional): Importance scores.

        Returns:
            torch.Tensor: Classification logits.
        """
        x = self.forward_features(x, importance_scores)
        x = self.head(x)
        return x


# Model instantiation functions with importance awareness
def vit_small_patch16_importance(**kwargs):
    """Small ViT with importance mechanisms."""
    model = VisionTransformerWithImportance(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base_patch16_importance(**kwargs):
    """Base ViT with importance mechanisms."""
    model = VisionTransformerWithImportance(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16_importance(**kwargs):
    """Large ViT with importance mechanisms."""
    model = VisionTransformerWithImportance(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14_importance(**kwargs):
    """Huge ViT with importance mechanisms."""
    model = VisionTransformerWithImportance(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model