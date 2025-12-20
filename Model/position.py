import torch

device = "cuda" if torch.cuda.is_available() else "mps"


def get_2D_sincos_position_embedding(embedding_dim, gridsize):

    grid_h = torch.arange(gridsize, dtype=torch.float32, device=device)
    grid_w = torch.arange(gridsize, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # (2, gridsize * gridsize)

    pos_embedding_h = get_1D_sincos_position_embedding(embedding_dim // 2, grid[0])
    pos_embedding_w = get_1D_sincos_position_embedding(embedding_dim // 2, grid[1])
    pos_hw = torch.cat([pos_embedding_h, pos_embedding_w], dim=1)  # (G^2, D)

    cls_pad = torch.zeros([1, embedding_dim])

    # prepend CLS as a separate token
    pos_embeddings = torch.cat([cls_pad, pos_hw], dim=0)  # (G^2 + 1, D)

    return pos_embeddings.unsqueeze(0)


def get_1D_sincos_position_embedding(embedding_dim, pos):
    frequencies = torch.arange(embedding_dim // 2, dtype=torch.float32, device=device)
    frequencies /= embedding_dim / 2.0
    frequencies = 1.0 / (10000**frequencies)

    pos = pos.reshape(-1)

    out = torch.einsum("m, d->md", pos, frequencies)

    sine_frequencies = torch.sin(out)
    cosine_frequencies = torch.cos(out)

    pos_embeddings = torch.cat([sine_frequencies, cosine_frequencies], dim=1)
    return pos_embeddings
