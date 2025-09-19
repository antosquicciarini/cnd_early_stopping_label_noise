import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, input_channels=3, num_classes=10, dim=64, depth=4, heads=4, mlp_dim=128):
        super(LightweightViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels  # Include input_channels
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_classes = num_classes

        # Ensure image dimensions are divisible by the patch size
        assert image_size % patch_size == 0, "Image size must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = input_channels * (patch_size ** 2)  # Adjusted for input_channels

        # Patch embedding layer
        self.patch_embedding = nn.Linear(self.patch_dim, dim)

        # Position embeddings
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)
        ])

        # MLP head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, indx, return_intermediates=False, CND_reg_only_last_layer=False, apply_mask=False,  neuron_indexes=[0], layer_indexes=[]):
        # Split image into patches
        batch_size = x.shape[0]
        patches = self.image_to_patches(x)
        patches = self.patch_embedding(patches)

        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat([cls_tokens, patches], dim=1)
        x += self.position_embedding

        # Collect intermediate activations if required
        intermediates = []

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x, q, k, v = block(x)
            if return_intermediates:
                # torch.cat([q.detach(), k.detach(), v.detach()], dim=0)
                # intermediates['queries'].append(q.detach())
                # intermediates['keys'].append(k.detach())
                # intermediates['values'].append(v.detach())

                # Permute to [Batch, Embedding Dim, Sequence Length]
                x_permuted = x.permute(0, 2, 1)  # Shape: [256, 64, 65]
                # Apply 1D average pooling over the sequence length (dim=2 here)
                pooled = F.avg_pool1d(x_permuted, kernel_size=x_permuted.shape[2])  # Shape: [256, 64, 1]
                # Squeeze the last dimension to get [256, 64]
                pooled = pooled.squeeze(-1)

                intermediates.append(pooled.detach())

        # Classification head
        x = self.mlp_head(x[:, 0])

        if return_intermediates:
            return x, torch.cat(intermediates, dim=1)

        return x

    def image_to_patches(self, x):
        # Reshape image into patches
        batch_size, channels, height, width = x.shape
        assert channels == self.input_channels, f"Expected input channels {self.input_channels}, but got {channels}"
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, num_patches_x, num_patches_y, C, patch_h, patch_w]
        x = x.view(batch_size, -1, self.patch_dim)  # Flatten patches
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, q, k, v = self.attention(self.norm1(x))
        x = x + attn_output

        # MLP block
        x = x + self.mlp(self.norm2(x))

        return x, q, k, v


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2),
            qkv
        )

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Linear projection
        out = self.fc_out(out)
        return out, q, k, v
    


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LightweightViT_2(nn.Module):
    def __init__(self, image_size=32, patch_size=4, input_channels=3, num_classes=10, dim=64, depth=4, heads=4, mlp_dim=128, dropout=0.1):
        super(LightweightViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.dim = dim

        # Ensure image dimensions are divisible by the patch size
        assert image_size % patch_size == 0, "Image size must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = input_channels * (patch_size ** 2)

        # Spatial Patch Tokenization (SPT)
        self.to_patch_embedding = SPT(dim=dim, patch_size=patch_size, channels=input_channels)

        # Positional Embedding and Class Token
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer Layers
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)

        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, return_intermediates=False):
        batch_size = x.shape[0]

        # Tokenize image into patches with SPT
        x = self.to_patch_embedding(x)

        # Add class token and positional embeddings
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Pass through the Transformer
        x = self.transformer(x)

        # Classification
        x = x[:, 0]  # Take the class token
        return self.mlp_head(x)


class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim=1)
        return self.to_patch_tokens(x_with_shifts)


class LSA(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, LSA(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)