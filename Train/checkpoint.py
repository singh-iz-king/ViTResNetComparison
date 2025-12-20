import os
import torch


def load_checkpoint(model, optimizer, filename, map_location="cpu"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location=map_location)

        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        metrics = checkpoint.get("metrics", {})
        print(f"Resuming from epoch {start_epoch}, previous metrics: {metrics}")
        return start_epoch
    else:
        print(f"No checkpoint found at {filename}. Starting from epoch 0.")
        return 0


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")
