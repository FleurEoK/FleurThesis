# models/models_semaim_importance.py
"""
Importance-Aware SemAIM based on the original AimViT architecture
Integrates semantic importance from detection overlap counts
"""

import math
from functools import partial
import json
import os

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from util.blocks import GaussianConv2d
from util.blocks import Block_SelfMask, Block_SelfCrossMask


class ImportanceAwareBlock_SelfMask(Block_SelfMask):
    """
    Extended Block_SelfMask with importance-aware attention bias
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learnable scaling for importance bias
        self.importance_bias_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x, mask=None, importance_bias=None):
        """
        Forward with optional importance bias
        Args:
            x: input tokens
            mask: attention mask from permutation
            importance_bias: [B, 1, N+1, N+1] importance attention bias
        """
        # If importance bias provided, add it to the mask
        if importance_bias is not None and mask is not None:
            # Scale and combine with existing mask
            combined_mask = mask + importance_bias * self.importance_bias_scale
            return super().forward(x, mask=combined_mask)
        return super().forward(x, mask=mask)


class ImportanceAwareBlock_SelfCrossMask(Block_SelfCrossMask):
    """
    Extended Block_SelfCrossMask with importance-aware attention bias
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.importance_bias_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x, context, mask=None, importance_bias=None):
        """
        Forward with optional importance bias
        """
        if importance_bias is not None and mask is not None:
            combined_mask = mask + importance_bias * self.importance_bias_scale
            return super().forward(x, context, mask=combined_mask)
        return super().forward(x, context, mask=mask)


class AimViT_Importance(nn.Module):
    """
    Importance-Aware AimViT
    Extends original AimViT with semantic importance weighting from detection overlap counts
    
    Key additions:
    - Importance-weighted position encoding
    - Importance-based attention bias
    - Support for loading importance scores from JSON
    """

    def __init__(self,
                 # vision transformer backbone
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, drop_path_rate=0., out_dim=768,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 # aim
                 permutation_type='center2out', attention_type='cls',
                 # decoder
                 query_depth=12, share_weight=False,
                 prediction_head_type='MLP',
                 # loss function
                 gaussian_kernel_size=None, gaussian_sigma=None,
                 loss_type='L2', predict_feature='none', norm_pix_loss=True,
                 # importance parameters (NEW)
                 use_importance_bias=True,
                 use_importance_pe=True,
                 importance_json_path=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Importance parameters
        self.use_importance_bias = use_importance_bias
        self.use_importance_pe = use_importance_pe

        # patch embedding layer
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_patches = num_patches

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # position embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Importance-aware position encoding (NEW)
        if use_importance_pe:
            self.importance_pe_scale = nn.Parameter(torch.ones(1))
        
        if use_importance_bias:
            self.importance_bias_scale = nn.Parameter(torch.ones(1))

        # encoder: uses importance-aware blocks
        if use_importance_bias:
            self.blocks = nn.ModuleList([
                ImportanceAwareBlock_SelfMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                                              norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                Block_SelfMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                              norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(depth)])

        # decoder: uses importance-aware blocks
        if share_weight:
            self.query_blocks = self.blocks
        else:
            if use_importance_bias:
                self.query_blocks = nn.ModuleList([
                    ImportanceAwareBlock_SelfCrossMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                                                       norm_layer=norm_layer, drop_path=dpr[i])
                    for i in range(query_depth)])
            else:
                self.query_blocks = nn.ModuleList([
                    Block_SelfCrossMask(embed_dim, num_heads, mlp_ratio, qkv_bias=True, 
                                       norm_layer=norm_layer, drop_path=dpr[i])
                    for i in range(query_depth)])
        
        self.depth = depth
        self.step = depth // query_depth

        # permutation type
        self.permutation_type = permutation_type

        # prediction head
        self.norm = norm_layer(embed_dim)
        self.predict_feature = predict_feature
        self.attention_type = attention_type
        
        if prediction_head_type == 'LINEAR':
            if predict_feature == 'none':
                self.prediction_head = nn.Linear(embed_dim, patch_size ** 2 * 3)
            else:
                rec_dim = out_dim if predict_feature == 'clip' else embed_dim
                self.prediction_head = nn.Linear(embed_dim, rec_dim)
        elif prediction_head_type == 'MLP':
            if predict_feature == 'none':
                self.prediction_head = Mlp(embed_dim, int(embed_dim * mlp_ratio), patch_size ** 2 * 3)
            else:
                rec_dim = out_dim if predict_feature == 'clip' else embed_dim
                self.prediction_head = Mlp(embed_dim, int(embed_dim * mlp_ratio), rec_dim)

        # loss parameters
        self.loss_type = loss_type
        self.norm_pix_loss = norm_pix_loss
        if gaussian_kernel_size is not None and gaussian_sigma is not None and self.predict_feature == 'none':
            self.gaussian_blur = GaussianConv2d(3, gaussian_kernel_size, gaussian_sigma)
        else:
            self.gaussian_blur = nn.Identity()

        # split matrix for guided center permutation
        num_patch = img_size // patch_size
        split_matrix = torch.zeros((14, 2, 4))
        split_matrix[0, :, :] = torch.tensor([[0, 0, 0, 0], [2, 6, 10, 13]])
        split_matrix[1, :, :] = torch.tensor([[0, 0, 0, 0], [2, 6, 10, 13]])
        split_matrix[2, :, :] = torch.tensor([[0, 0, 0, 0], [2, 6, 10, 13]])
        split_matrix[3, :, :] = torch.tensor([[2, 0, 0, 0], [4, 6, 10, 13]])
        split_matrix[4, :, :] = torch.tensor([[3, 1, 0, 0], [5, 7, 10, 13]])
        split_matrix[5, :, :] = torch.tensor([[4, 2, 0, 0], [6, 8, 10, 13]])
        split_matrix[6, :, :] = torch.tensor([[5, 3, 1, 0], [7, 9, 11, 13]])
        split_matrix[7, :, :] = torch.tensor([[6, 4, 2, 0], [8, 10, 12, 13]])
        split_matrix[8, :, :] = torch.tensor([[7, 5, 3, 0], [9, 11, 13, 13]])
        split_matrix[9, :, :] = torch.tensor([[8, 6, 3, 0], [10, 12, 13, 13]])
        split_matrix[10, :, :] = torch.tensor([[9, 7, 3, 0], [11, 13, 13, 13]])
        split_matrix[11, :, :] = torch.tensor([[11, 7, 3, 0], [13, 13, 13, 13]])
        split_matrix[12, :, :] = torch.tensor([[11, 7, 3, 0], [13, 13, 13, 13]])
        split_matrix[13, :, :] = torch.tensor([[11, 7, 3, 0], [13, 13, 13, 13]])
        self.split_matrix = split_matrix

        # coordinates for patches (row, col)
        coordinates = torch.zeros((num_patches, 2))
        for i in range(num_patch):
            for j in range(num_patch):
                coordinates[i*num_patch+j, 0] = i  # row
                coordinates[i*num_patch+j, 1] = j  # col
        self.coordinates = coordinates.unsqueeze(0)

        # Load importance scores if provided (NEW)
        self.importance_data = None
        if importance_json_path and os.path.exists(importance_json_path):
            with open(importance_json_path, 'r') as f:
                self.importance_data = json.load(f)
            print(f"Loaded importance scores for {len(self.importance_data)} images from {importance_json_path}")

        # initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                           int(self.patch_embed.num_patches ** .5),
                                           cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize cls_token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # ============ Importance-aware methods (NEW) ============
    
    def get_importance_scores(self, image_names=None, batch_size=None):
        """
        Get importance scores for the batch
        Args:
            image_names: list of image names (if available)
            batch_size: batch size (if image_names not available)
        Returns:
            importance_scores: [B, num_patches] tensor
        """
        if image_names is not None and self.importance_data is not None:
            scores_list = []
            for img_name in image_names:
                if img_name in self.importance_data:
                    scores = self.importance_data[img_name]['importance_scores']
                    scores_list.append(scores)
                else:
                    # Default: uniform importance
                    scores_list.append([1.0] * self.num_patches)
            importance_scores = torch.tensor(scores_list, dtype=torch.float32)
        else:
            # Default: uniform importance for all tokens
            B = batch_size if batch_size is not None else 1
            importance_scores = torch.ones(B, self.num_patches, dtype=torch.float32)
        
        return importance_scores
    
    def normalize_importance_scores(self, importance_scores):
        """Normalize importance scores to [0, 1] range"""
        max_vals = importance_scores.max(dim=1, keepdim=True)[0]
        normalized = importance_scores / (max_vals + 1e-8)
        return normalized
    
    def create_importance_position_encoding(self, importance_scores):
        """
        Create importance-weighted position encoding
        Args:
            importance_scores: [B, num_patches] normalized (0-1)
        Returns:
            weighted_pe: [B, num_patches, embed_dim]
        """
        B = importance_scores.shape[0]
        
        # Get standard position embeddings (without CLS)
        pos_embed_patches = self.pos_embed[:, 1:, :].expand(B, -1, -1)  # [B, N, D]
        
        # Weight by importance
        importance_weights = importance_scores.unsqueeze(-1)  # [B, N, 1]
        weighted_pe = pos_embed_patches * importance_weights * self.importance_pe_scale
        
        return weighted_pe
    
    def create_importance_attention_bias(self, importance_scores):
        """
        Create attention bias matrix from importance scores
        Args:
            importance_scores: [B, N] normalized (0-1)
        Returns:
            bias: [B, 1, N+1, N+1] including CLS token
        """
        B, N = importance_scores.shape
        
        # Add CLS token (always importance = 1.0)
        cls_importance = torch.ones(B, 1, device=importance_scores.device)
        full_importance = torch.cat([cls_importance, importance_scores], dim=1)  # [B, N+1]
        
        # Create bias matrix: high importance tokens should receive more attention
        query_importance = full_importance.unsqueeze(2)  # [B, N+1, 1]
        key_importance = full_importance.unsqueeze(1)    # [B, 1, N+1]
        
        bias = query_importance * key_importance  # [B, N+1, N+1]
        
        # Convert to log space (added to attention logits)
        bias = torch.log(bias + 1e-10) * self.importance_bias_scale
        
        return bias.unsqueeze(1)  # [B, 1, N+1, N+1]
    
    def generate_importance_based_permutation(self, importance_scores):
        """
        Generate permutation based on importance scores
        High importance = processed first (lower permutation value)
        Args:
            importance_scores: [B, N]
        Returns:
            permutation: [B, N]
        """
        # Invert importance (high importance = low permutation value)
        permutation = 1.0 - importance_scores
        
        # Add small random noise for stability
        permutation = permutation + torch.rand_like(permutation) * 1e-6
        
        return permutation

    # ============ Original permutation methods ============
    
    def generate_raster_permutation(self, N, L):
        """Generate raster permutation"""
        width = int(L ** 0.5)
        permutation = torch.zeros((N, width, width))

        init_value = 0
        odd_row = torch.tensor([13 - i for i in range(width)])
        even_row = torch.tensor([i for i in range(width)])
        for i in range(width):
            if i % 2 == 0:
                permutation[:, i, :] = even_row + init_value
            else:
                permutation[:, i, :] = odd_row + init_value
            init_value += width

        permutation = permutation.reshape(N, L)
        return permutation

    def generate_center_permutation(self, N, L, center_first=True):
        """Generate center-out permutation"""
        width = int(L ** 0.5)
        half_width = width // 2
        permutation = torch.rand((N, width, width))

        if center_first:
            permutation[:, half_width-1:half_width+1, half_width-1:half_width+1] -= 1
            permutation[:, half_width-3:half_width+3, half_width-3:half_width+3] -= 1
            permutation[:, half_width-5:half_width+5, half_width-5:half_width+5] -= 1
        else:
            permutation[:, half_width-1:half_width+1, half_width-1:half_width+1] += 1
            permutation[:, half_width-3:half_width+3, half_width-3:half_width+3] += 1
            permutation[:, half_width-5:half_width+5, half_width-5:half_width+5] += 1

        permutation = permutation.reshape(N, L)
        return permutation

    def generate_stochastic_center_permutation(self, N, L):
        """Generate stochastic center permutation"""
        width = int(L ** 0.5)
        permutation = torch.rand((N, width, width))

        center_row, center_col = torch.rand((N)) * (width - 1), torch.rand((N)) * (width - 1)

        for i in range(N):
            row_split = self.split_matrix[int(center_row[i]), :, :]
            col_split = self.split_matrix[int(center_col[i]), :, :]
            for j in range(3):
                permutation[i, int(row_split[0][j]):int(row_split[1][j]), 
                          int(col_split[0][j]):int(col_split[1][j])] -= 1

        permutation = permutation.reshape(N, L)
        return permutation
    
    def generate_guided_center_permutation(self, attention_maps):
        """Generate attention guided center permutation"""
        N, L = attention_maps.shape
        width = int(L ** 0.5)
        permutation = torch.rand((N, width, width))

        _, max_index = torch.max(attention_maps, dim=-1)
        center_row, center_col = max_index // width, max_index % width

        for i in range(N):
            row_split = self.split_matrix[center_row[i], :, :]
            col_split = self.split_matrix[center_col[i], :, :]
            for j in range(3):
                permutation[i, int(row_split[0][j]):int(row_split[1][j]), 
                          int(col_split[0][j]):int(col_split[1][j])] -= 1

        permutation = permutation.reshape(N, L)
        return permutation

    def generate_attention_distance_center_permutation(self, attention_maps):
        """Generate attention guided gaussian center permutation"""
        N, L = attention_maps.shape
        width = int(L ** 0.5)

        _, max_index = torch.max(attention_maps, dim=-1)
        center_row, center_col = max_index // width, max_index % width

        self.coordinates = self.coordinates.cuda()
        permutation = (self.coordinates[:, :, 0] - center_row.unsqueeze(1)) ** 2 + \
                     (self.coordinates[:, :, 1] - center_col.unsqueeze(1)) ** 2
        permutation = permutation ** 0.5
        permutation += torch.rand(N, L).cuda() * 1e-3

        return permutation

    def generate_attention_mask(self, x, attention_maps=None, importance_scores=None):
        """
        Generate permutation mask with optional importance weighting
        Args:
            x: patch embeddings [B, N, D]
            attention_maps: attention maps for guided permutation
            importance_scores: [B, N] importance scores (NEW)
        """
        N, L, D = x.shape

        # Generate permutation based on type
        if self.permutation_type == 'zigzag':
            permutation = torch.tensor([i for i in range(L)]).repeat(N, 1).cuda()
        elif self.permutation_type == 'raster':
            permutation = self.generate_raster_permutation(N, L).to(self.device)
        elif self.permutation_type == 'stochastic':
            permutation = torch.rand(N, L, device=x.device)
        elif self.permutation_type == 'stochastic_center':
            permutation = self.generate_stochastic_center_permutation(N, L).cuda()
        elif self.permutation_type == 'center2out':
            permutation = self.generate_center_permutation(N, L).cuda()
        elif self.permutation_type == 'attention':
            assert attention_maps is not None
            assert attention_maps.shape[1] == L
            permutation = 1 - attention_maps
        elif self.permutation_type == 'attention_guided':
            assert attention_maps is not None
            assert attention_maps.shape[1] == L
            permutation = self.generate_guided_center_permutation(attention_maps).cuda()
        elif self.permutation_type == 'attention_center':
            assert attention_maps is not None
            assert attention_maps.shape[1] == L
            permutation = self.generate_attention_distance_center_permutation(attention_maps)
        elif self.permutation_type == 'importance':  # NEW
            assert importance_scores is not None
            permutation = self.generate_importance_based_permutation(importance_scores)
        elif self.permutation_type == 'spatial_importance':  # NEW - spatial order with importance bias
            # Keep spatial order (raster), importance used only as attention bias
            permutation = torch.arange(L, device=x.device).float().unsqueeze(0).repeat(N, 1)
        else:
            print(f"Not supported permutation type: {self.permutation_type}")
            permutation = torch.rand(N, L, device=x.device)

        # Create content mask and query mask
        full_mask = torch.full((N, L, L), -math.inf, device=x.device)
        no_mask = torch.zeros((N, L, L), device=x.device)
        mask_h = torch.where(permutation.unsqueeze(-1) < permutation.unsqueeze(1), full_mask, no_mask)
        mask_g = torch.where(permutation.unsqueeze(-1) <= permutation.unsqueeze(1), full_mask, no_mask)

        # Consider cls_token
        top_padding = torch.full((N, 1, L), -math.inf, device=x.device)
        left_padding = torch.zeros((N, L + 1, 1), device=x.device)
        mask_h = torch.cat((top_padding, mask_h), dim=1)
        mask_h = torch.cat((left_padding, mask_h), dim=2)
        mask_g = torch.cat((top_padding, mask_g), dim=1)
        mask_g = torch.cat((left_padding, mask_g), dim=2)
        
        return mask_h.unsqueeze(1), mask_g.unsqueeze(1), permutation

    def forward_aim(self, x, attention_maps=None, importance_scores=None, image_names=None):
        """
        Forward pass with importance awareness
        Args:
            x: input images [B, 3, H, W]
            attention_maps: attention maps for guided permutation
            importance_scores: [B, N] pre-computed importance scores (optional)
            image_names: list of image names for loading importance scores
        """
        # Embed patches
        x = self.patch_embed(x)
        B = x.shape[0]

        # Get importance scores if using importance-aware features
        if self.use_importance_bias or self.use_importance_pe:
            if importance_scores is None:
                importance_scores = self.get_importance_scores(image_names, batch_size=B)
            importance_scores = importance_scores.to(x.device)
            normalized_importance = self.normalize_importance_scores(importance_scores)
        else:
            normalized_importance = None

        # Generate attention masks
        mask_h, mask_g, permutation = self.generate_attention_mask(x, attention_maps, normalized_importance)

        # Create importance attention bias if enabled
        importance_bias = None
        if self.use_importance_bias and normalized_importance is not None:
            importance_bias = self.create_importance_attention_bias(normalized_importance)

        # Add position embeddings
        if self.use_importance_pe and normalized_importance is not None:
            # Add standard position embedding
            x = x + self.pos_embed[:, 1:, :]
            # Add importance-weighted position encoding
            importance_pe = self.create_importance_position_encoding(normalized_importance)
            x = x + importance_pe
        else:
            # Standard position embedding only
            x = x + self.pos_embed[:, 1:, :]

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Encoder with importance bias
        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)
        
        for i in range(self.depth):
            if self.use_importance_bias and isinstance(self.blocks[i], ImportanceAwareBlock_SelfMask):
                h = self.blocks[i](h, mask=mask_h, importance_bias=importance_bias)
            else:
                h = self.blocks[i](h, mask=mask_h)
            
            if (i + 1) % self.step == 0:
                if self.use_importance_bias and isinstance(self.query_blocks[i // self.step], ImportanceAwareBlock_SelfCrossMask):
                    g = self.query_blocks[i // self.step](g, h, mask=mask_g, importance_bias=importance_bias)
                else:
                    g = self.query_blocks[i // self.step](g, h, mask=mask_g)
        
        g = self.norm(g)
        g = self.prediction_head(g)

        return g, permutation

    def forward_aim_no_mask(self, x, attention_maps=None, importance_scores=None, image_names=None):
        """Forward without masking (for ablation)"""
        x = self.patch_embed(x)
        B = x.shape[0]

        # Get importance scores
        if self.use_importance_bias or self.use_importance_pe:
            if importance_scores is None:
                importance_scores = self.get_importance_scores(image_names, batch_size=B)
            importance_scores = importance_scores.to(x.device)
            normalized_importance = self.normalize_importance_scores(importance_scores)
        else:
            normalized_importance = None

        mask_h, mask_g, permutation = self.generate_attention_mask(x, attention_maps, normalized_importance)

        # Add position embeddings with importance
        if self.use_importance_pe and normalized_importance is not None:
            x = x + self.pos_embed[:, 1:, :]
            importance_pe = self.create_importance_position_encoding(normalized_importance)
            x = x + importance_pe
        else:
            x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)
        
        for i in range(self.depth):
            h = self.blocks[i](h)
            if (i + 1) % self.step == 0:
                g = self.query_blocks[i // self.step](g, h)
        
        g = self.norm(g)
        g = self.prediction_head(g)

        return g, permutation

    # ============ Keep all original methods ============
    
    def forward_encoder(self, x):
        """Original forward encoder"""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for i in range(len(self.blocks)-1):
            x = self.blocks[i](x)
        
        self_attention = self.blocks[len(self.blocks)-1](x, return_attention=True)
        self_attention = torch.mean(self_attention, dim=1)[:, 0, 1:]

        x = self.blocks[len(self.blocks)-1](x)
        x = self.norm(x)

        if self.attention_type == 'gap':
            feature_attention = self.calculate_attention_gap(x)
        else:
            feature_attention = self.calculate_attention_cls(x)

        return x, feature_attention, self_attention

    def calculate_attention_cls(self, tokens):
        """Original attention calculation"""
        tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        attention = torch.sum(tokens[:, 0, :].unsqueeze(1) * tokens[:, 1:, :], dim=-1)
        attention = attention.softmax(dim=1)
        return attention

    def calculate_attention_gap(self, tokens):
        """Original GAP attention calculation"""
        pth_gap = torch.mean(tokens[:, 1:, :], dim=1, keepdim=True)
        pth_gap = torch.nn.functional.normalize(pth_gap, p=2, dim=-1)
        tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
        attention = torch.sum(pth_gap * tokens[:, 1:, :], dim=-1)
        attention = attention.softmax(dim=1)
        return attention

    def forward_pixel_loss(self, imgs, pred):
        """Original pixel loss"""
        imgs = self.gaussian_blur(imgs)
        target = self.patchify(imgs)
        pred = pred[:, 1:, :]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5
        if self.loss_type == 'L1':
            loss = (pred - target).abs()
        elif self.loss_type == 'L2':
            loss = (pred - target) ** 2
        return loss.mean(), loss.mean(dim=-1)

    def forward_feature_loss(self, feature, pred):
        """Original feature loss"""
        feature = feature[:, 1:, :]
        pred = pred[:, 1:, :]
        feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        loss = ((pred - feature) ** 2).sum(dim=-1)
        return loss.mean(), loss

    def forward(self, imgs, tokens=None, attention_maps=None, forward_encoder=False, 
                importance_scores=None, image_names=None):
        """
        Main forward pass with importance awareness
        Args:
            imgs: input images [B, 3, H, W]
            tokens: teacher tokens (for feature prediction)
            attention_maps: attention maps for guided permutation
            forward_encoder: whether to forward encoder only
            importance_scores: [B, num_patches] pre-computed importance scores (NEW)
            image_names: list of image names for loading importance (NEW)
        """
        if forward_encoder:
            enc_tokens, feature_attention, self_attention = self.forward_encoder(imgs)
            return enc_tokens, feature_attention, self_attention

        # Forward with importance
        pred, permutation = self.forward_aim(imgs, attention_maps, importance_scores, image_names)

        if self.predict_feature == 'none':
            loss, loss_map = self.forward_pixel_loss(imgs, pred)
        else:
            assert tokens is not None
            loss, loss_map = self.forward_feature_loss(tokens, pred)
        
        return loss, permutation, loss_map

    def forward_for_visilization(self, imgs, attention_maps=None, importance_scores=None, image_names=None):
        """Forward for visualization with importance"""
        pred, permutation = self.forward_aim(imgs, attention_maps, importance_scores, image_names)
        loss, loss_map = self.forward_pixel_loss(imgs, pred)
        imgs_blur = self.gaussian_blur(imgs)
        return loss, permutation, pred, imgs_blur

    # ============ Generation methods (keep original) ============
    
    def generate_raster_permutation_for_generate(self, N, L):
        """Generate raster permutation for generation"""
        width = int(L ** 0.5)
        permutation = torch.zeros((N, width, width))

        init_value = 0
        odd_row = torch.tensor([13 - i for i in range(width)])
        even_row = torch.tensor([i for i in range(width)])
        for i in range(width):
            if i < width // 2:
                continue
            if i % 2 == 0:
                permutation[:, i, :] = even_row + init_value
            else:
                permutation[:, i, :] = odd_row + init_value
            init_value += width

        permutation = permutation.reshape(N, L)
        return permutation

    def generate_center_permutation_for_generate(self, N, L):
        """Generate center-out permutation for generation"""
        width = int(L ** 0.5)
        half_width = width // 2
        permutation = torch.zeros((N, width, width))

        permutation[:, half_width-3:half_width+3, half_width-3:half_width+3] -= 1
        permutation[:, half_width-5:half_width+5, half_width-5:half_width+5] -= 1
        permutation[:, half_width-7:half_width+7, half_width-7:half_width+7] -= 1

        permutation = permutation.reshape(N, L)
        return permutation

    def generate_attention_mask_for_generate(self, x):
        """Generate permutation mask for generation"""
        N, L, D = x.shape

        if self.permutation_type == 'raster':
            permutation = self.generate_raster_permutation_for_generate(N, L).cuda()
        elif self.permutation_type == 'center2out':
            permutation = self.generate_center_permutation_for_generate(N, L).cuda()
        else:
            print("Not supported permutation type!")
            permutation = torch.zeros(N, L).cuda()

        full_mask = torch.full((N, L, L), -math.inf, device=x.device)
        no_mask = torch.zeros((N, L, L), device=x.device)
        mask_g = torch.where(permutation.unsqueeze(-1) < permutation.unsqueeze(1), full_mask, no_mask)

        top_padding = torch.zeros((N, 1, L), device=x.device)
        left_padding = torch.zeros((N, L + 1, 1), device=x.device)
        mask_g = torch.cat((top_padding, mask_g), dim=1)
        mask_g = torch.cat((left_padding, mask_g), dim=2)
        return mask_g.unsqueeze(1), permutation

    def forward_aim_for_generate(self, x):
        """Forward for generation"""
        x = self.patch_embed(x)
        mask_g, permutation = self.generate_attention_mask_for_generate(x)

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        h = x
        g = self.pos_embed.expand(x.shape[0], -1, -1)
        for i in range(self.depth):
            h = self.blocks[i](h)
            if (i + 1) % self.step == 0:
                g = self.query_blocks[i // self.step](g, h, mask=mask_g)
        g = self.norm(g)
        g = self.prediction_head(g)

        return g, permutation


# ============ Model factory functions ============

def aim_base_importance(**kwargs):
    """Base model with importance awareness"""
    return AimViT_Importance(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        mlp_ratio=4, **kwargs
    )


def aim_large_importance(**kwargs):
    """Large model with importance awareness"""
    return AimViT_Importance(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, **kwargs
    )


def aim_huge_importance(**kwargs):
    """Huge model with importance awareness"""
    return AimViT_Importance(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, 
        mlp_ratio=4, **kwargs
    )


if __name__ == '__main__':
    # Test the importance-aware model
    torch.manual_seed(2023)
    
    print("Testing Importance-Aware AimViT")
    print("=" * 60)
    
    # Create model
    model = aim_base_importance(
        img_size=224,
        norm_pix_loss=False,
        permutation_type='spatial_importance',  # NEW permutation type
        prediction_head_type='MLP',
        loss_type='L2',
        query_depth=12,
        share_weight=False,
        gaussian_kernel_size=9,
        gaussian_sigma=1,
        use_importance_bias=True,
        use_importance_pe=True,
        importance_json_path='/home/20204130/Falcon/Polygon/results_vis/training_data_output/training_data.json'
    )
    
    model.eval()
    
    # Test inputs
    batch_size = 2
    x = torch.rand(batch_size, 3, 224, 224)
    
    # Test with dummy importance scores
    importance_scores = torch.rand(batch_size, 196)  # 14x14 = 196 patches
    
    print(f"Input shape: {x.shape}")
    print(f"Importance scores shape: {importance_scores.shape}")
    
    # Forward pass
    with torch.no_grad():
        loss, permutation, loss_map = model(x, importance_scores=importance_scores)
    
    print(f"\nForward pass successful!")
    print(f"Loss: {loss.item():.4f}")
    print(f"Permutation shape: {permutation.shape}")
    print(f"Loss map shape: {loss_map.shape}")
    
    # Test without importance (ablation)
    print("\n" + "=" * 60)
    print("Testing without importance (baseline)")
    model_baseline = aim_base_importance(
        img_size=224,
        norm_pix_loss=False,
        permutation_type='center2out',  # Original permutation
        prediction_head_type='MLP',
        loss_type='L2',
        query_depth=12,
        share_weight=False,
        use_importance_bias=False,  # Disabled
        use_importance_pe=False      # Disabled
    )
    
    model_baseline.eval()
    with torch.no_grad():
        loss_baseline, _, _ = model_baseline(x)
    
    print(f"Baseline loss: {loss_baseline.item():.4f}")
    print(f"Importance-aware loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)