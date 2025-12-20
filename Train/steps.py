import torch
import torch.nn.functional as F


def mae_step(model, batch, device):

    batch = batch.to(device)

    reconstruction, mask = model(batch)

    loss = (batch - reconstruction) ** 2
    loss = loss.mean(dim=-1)  # (B, N)

    loss = (loss * mask).sum() / mask.sum()  # (1)

    metrics = {"loss": loss.item()}

    return loss, metrics


def simclr_step(model, batch, device, tau=0.2):
    im1, im2 = batch
    im1, im2 = im1.to(device), im2.to(device)
    B = im1.shape[0]

    z1 = model(im1)  # (B, D)
    z2 = model(im2)  # (B, D)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = (z @ z.T) / tau  # (2B, 2B)

    # remove self-similarity
    sim = sim.masked_fill(torch.eye(2 * B, device=device, dtype=torch.bool), -1e9)

    # positives: i<->i+B
    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device)  # (2B,)

    loss = F.cross_entropy(sim, pos)
    metrics = {"loss": loss.item()}
    return loss, metrics


def finetune_step(model, batch, device):

    imgs, labels = batch
    imgs, labels = imgs.to(device), labels.to(device)

    predicted_logits = model(imgs)

    loss = F.cross_entropy(predicted_logits, labels)

    predicted_labels = torch.argmax(predicted_logits, dim=1)

    acc = (predicted_labels == labels).float().mean().item()

    metrics = {"loss": loss.item(), "accuracy": acc}

    return loss, metrics
