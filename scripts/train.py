import torch
import argparse
import os
import yaml

from Train.builder.model import build_model
from Train.builder.optimizer import build_optimizers
from Train.builder.loader import build_loaders
from Train.builder.step import build_steps
from Train.builder.eval import build_evals
from Train.engine import train_epochs


def pick_device(prefer):
    prefer = (prefer or "").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("mps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to Yaml Config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = pick_device(cfg.get("device", {}).get("prefer", "cuda"))
    out_dir = cfg["run"]["out_dir"]
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

    seed = cfg.get("run", {}).get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    train_loader, val_loader = build_loaders(cfg)

    model = build_model(cfg).to(device)
    optimizer = build_optimizers(cfg, model)
    step_fn = build_steps(cfg)
    eval_fn = build_evals(cfg)

    t = cfg["train"]

    is_ssl = cfg["model"]["kind"] in ["mae", "simclr_vit"]

    train_epochs(
        model,
        train_loader,
        optimizer,
        step_fn,
        t["epochs"],
        t["ckpt_interval"],
        out_dir,
        device,
        val_loader,
        eval_fn,
        t["best_metric"],
        t["best_mode"],
        is_ssl,
    )


if __name__ == "__main__":
    main()
