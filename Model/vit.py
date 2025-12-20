import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.position import (
    get_2D_sincos_position_embedding,
)

device = "cuda" if torch.cuda.is_available() else "mps"


class PatchEmbedder(nn.Module):

    def __init__(self, embedding_dim, p, N):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.l1 = nn.Linear(p * p * 3, embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embedding_dim)))

        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        """
        Embedds image patches into vectors in latent space

        input:
            x (tensor) : (B, N, p * p * 3) patches to be embedded
        output:
            tensor : (B, N + 1, embedding_dim) -> embedded patches
        """

        B, N, _ = x.shape

        patches = self.l1(x)

        cls_expanded = self.cls_token.expand(B, -1, -1)

        patches_and_cls = torch.cat([cls_expanded, patches], dim=1)

        # Note we assume the patch width/height is equal to image token width/height

        return self.drop_out(patches_and_cls)


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.K = nn.Linear(embedding_dim, embedding_dim)
        self.Q = nn.Linear(embedding_dim, embedding_dim)
        self.V = nn.Linear(embedding_dim, embedding_dim)
        self.Projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Computes multiheaded self-attention

        Input:
            x (tensor) : (B, N + 1, embedding_dim) batched sequence

        Output:
            tensor : (B, N + 1, embedding_dim)
        """

        B, N_1, _ = x.shape

        keys = self.K(x)  # (B, N + 1, embedding_dim)
        queries = self.Q(x)  # (B, N + 1, embedding_dim)
        values = self.V(x)  # (B, N+1, embedding_dim)

        # Reshaping ks, qs, vs for attention computation #(B, N + 1, num_heads, embedding_dim / num_heads)
        keys = torch.reshape(
            keys, (B, N_1, self.num_heads, self.embedding_dim // self.num_heads)
        )

        queries = torch.reshape(
            queries, (B, N_1, self.num_heads, self.embedding_dim // self.num_heads)
        )

        values = torch.reshape(
            values, (B, N_1, self.num_heads, self.embedding_dim // self.num_heads)
        )

        # Switching order to group by head instead of by patch
        # -> (B, num_heads, N+1, embedding_dim / num_heads)

        keys = torch.swapaxes(keys, axis0=1, axis1=2).contiguous()
        queries = torch.swapaxes(queries, axis0=1, axis1=2).contiguous()
        values = torch.swapaxes(values, axis0=1, axis1=2).contiguous()

        similarities = torch.matmul(
            queries, torch.transpose(keys, axis0=2, axis1=3)
        )  # (B, num_heads, N+1, N+1)
        scaled_similarities = F.softmax(
            similarities / (self.embedding_dim / self.num_heads) ** 0.5, dim=-1
        )
        scaled_similarities = self.dropout(scaled_similarities)
        attention = torch.matmul(
            scaled_similarities, values
        )  # (B, num_heads, N+1, embedding_dim / num_heads)

        # Switching attention back to correct shape
        attention = torch.swapaxes(attention, axis0=1, axis2=2).contiguous()
        attention = torch.reshape(
            attention, shape=(B, N_1, self.embedding_dim)
        ).contiguous()

        return self.Projection(attention)


class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.msa = MultiheadedSelfAttention(6, embedding_dim=embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        """
        Computes one attention block in the ViT

        input:
            x (tensor) : (B, N + 1, embedding_dim)

        returns:
            tensor : (B, N+1, embedding_dim)
        """

        x_attention = x + self.msa(self.ln1(x))

        x_mlp = x_attention + self.mlp(self.ln2(x_attention))

        return x_mlp


class ViTEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.PatchEmbedder = PatchEmbedder(embedding_dim=embedding_dim, p=8, N=64)
        self.Attention = nn.Sequential(
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),  # 8 Attention Blocks
        )
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.PatchEmbedder.cls_token, std=0.02)

    def forward(self, x):
        """
        Computes forward pass for ViT

        inputs:
            x (tensor): (B, N, p*p*3) a batch of n p*p image patches

        returns:
            tenosr : (B, N+1, embedding_dim)
        """
        patches = self.PatchEmbedder(x)  # (B, N + 1, embedding_dim)

        patches_with_attention = self.Attention(patches)

        return patches_with_attention

    def _init_weights(self, m):
        """
        Custom weight initlization, because Transformers struggle with pytorch weight initilizaiton (vanishing/exploding gradients)

        input:
            m : type of layer
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MAEDecoder(nn.Module):
    def __init__(self, embedding_dim, p):
        super().__init__()

        self.mask = nn.Parameter(torch.randn(size=(1, 1, embedding_dim)))

        self.Attention = nn.Sequential(
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
            AttentionBlock(embedding_dim=embedding_dim),
        )

        self.Projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, p * p * 3),
        )

    def forward(self, x):
        """
        Decodes masked tokens using attention

        input:
            x (tensor) : (B, N + 1, embedding_dim), first 25% of N + 1 are real patches,
            rest are masked with 0

        returns:
            tensor : (B, (N + 1) * 0.75, p * p * 3)
        """
        attentioned_tokens = self.Attention(x)

        return self.Projection(attentioned_tokens)


class MAEWrapper(nn.Module):
    def __init__(self, embedding_dim, p):
        super().__init__()
        self.D = embedding_dim
        self.P = p
        self.Encoder = ViTEncoder(embedding_dim=embedding_dim)
        self.Decoder = MAEDecoder(embedding_dim=embedding_dim, p=p)

    def forward(self, x):
        """
        Computes forward pass for Masked Auto Encoder (SSL for ViT)

        input:
            x (tensor) : (B, number patches (N), p * p * 3)
        """

        # Encoding
        tokens = self.Encoder.PatchEmbedder(x)  # (B, N+1, D)

        tokens = tokens + get_2D_sincos_position_embedding(self.D, self.P).to(
            x.device
        )  # (B, N+1, D)

        cls_tokens = tokens[:, 0:1, :]  # (B, 1, D)

        encoder_tokens, mask, ids_restore = self.random_masking(
            tokens[:, 1:, :], mask_ratio=0.75
        )

        encoder_tokens = torch.cat(
            [cls_tokens, encoder_tokens], dim=1
        )  # (B, 0.25*N + 1, D)

        encoder_tokens = self.Encoder.Attention(encoder_tokens)  # (B, 0.25*n + 1, D)

        # Decoding
        cls_token = encoder_tokens[:, 0:1, :]
        x_patches = encoder_tokens[:, 1:, :]

        mask_tokens = self.Decoder.mask.repeat(
            encoder_tokens.shape[0], ids_restore.shape[1] - x_patches.shape[1], 1
        )
        # (B, mask_ratio*N, D)

        x_full = torch.cat([x_patches, mask_tokens], dim=1)

        # Unshuffle
        x_full = torch.gather(
            x_full, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.D), dim=1
        )
        x = torch.cat([cls_token, x_full], dim=1)

        x = x + get_2D_sincos_position_embedding(self.D, self.P).to(device=x.device)

        x = self.Decoder(x)  # (B, N+1, p*p*3)

        x_no_cls = x[:, 1:, :]  # (B, N, p*p*3)

        return x_no_cls, mask

    def random_masking(self, x, mask_ratio):
        B, N, D = x.shape  # Batch, Patches, Dim
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate binary mask for loss (0 keep, 1 remove)
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class SimCLRWrapper(nn.Module):
    def __init__(self, embedding_dim, p):
        super().__init__()
        self.D = embedding_dim
        self.P = p
        self.Encoder = ViTEncoder(embedding_dim=embedding_dim)
        self.Projection = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.GELU(), nn.Linear(128, 128)
        )

    def forward(self, x):
        """
        Computes forward pass for SimCLR SSL

        inputs:
            x (tensor) : (B, N, p*p*3)
        """
        tokens = self.Encoder.PatchEmbedder(x)  # (B, N + 1, embedding_dim)

        tokens = tokens + get_2D_sincos_position_embedding(self.D, self.P).to(
            x.device
        )  # (B, N+1, embedding_dim)

        tokens = self.Encoder.Attention(tokens)  # (B, N + 1, embedding_dim)

        cls_tokens = tokens[:, 0, :]  # (B, D)

        z = self.Projection(cls_tokens)

        z = F.normalize(z, dim=1)

        return z


class VitClassifierWrapper(nn.Module):
    def __init__(self, embedding_dim, p, num_classes):
        super().__init__()
        self.D = embedding_dim
        self.P = p
        self.Encoder = ViTEncoder(embedding_dim=embedding_dim)
        self.Classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, x):
        """
        Computes forward pass for VitClassifier

        inputs:
            x (tensor) : (B, N, p*p*3)

        returns:
            tensor : (B, num_classes)
        """
        tokens = self.Encoder.PatchEmbedder(x)  # (B, N + 1, embedding_dim)

        tokens = tokens + get_2D_sincos_position_embedding(self.D, self.P).to(
            x.device
        )  # (B, N+1, embedding_dim)

        tokens = self.Encoder.Attention(tokens)  # (B, N + 1, embedding_dim)

        cls_tokens = tokens[:, 0, :]  # (B, D)

        return self.Classifier(cls_tokens)
