import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedder(nn.Module):

    def __init__(self, embedding_dim, p, N):
        super().__init__()
        self.l1 = nn.Linear(p * p * 3, embedding_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.position_embedding = nn.Parameter(torch.randn(1, N + 1, embedding_dim))

        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        """
        Embedds image patches into vectors in latent space and adds position information

        input:
            x (tensor) : (B, N, p * p * 3) patches to be embedded
        output:
            tensor : (B, N + 1, embedding_dim) -> embedded patches
        """

        B, _, _ = x.shape

        patches = self.l1(x)

        cls_expanded = self.cls_token.expand(B, -1, -1)

        patches_and_cls = torch.cat([cls_expanded, patches], dim=1)

        patches_and_cls_with_position = patches_and_cls + self.position_embedding

        return self.drop_out(patches_and_cls_with_position)


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


class ViT(nn.Module):
    def __init__(self, embedding_dim, num_classes):
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
        self.ClassifierHead = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, num_classes)
        )
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.PatchEmbedder.cls_token, std=0.02)
        nn.init.trunc_normal_(self.PatchEmbedder.position_embedding, std=0.02)

    def forward(self, x):
        """
        Computes forward pass for ViT

        inputs:
            x (tensor): (B, N, p*p*3) a batch of n p*p image patches

        returns:
            tenosr : (B, num_classes)
        """
        patches = self.PatchEmbedder(x)  # (B, N + 1, embedding_dim)

        patches_with_attention = self.Attention(patches)

        cls = patches_with_attention[:, 0, :]

        return self.ClassifierHead(cls)

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
