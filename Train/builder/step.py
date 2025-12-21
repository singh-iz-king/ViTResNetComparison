import torch
from Train.steps import mae_step, simclr_step, finetune_step


def build_steps(cfg):
    m = cfg["model"]
    kind = m["kind"]

    if kind == "mae":
        return mae_step
    if kind == "vit_classifier":
        return finetune_step
    if kind == "simclr_vit":
        return simclr_step

    raise ValueError(f"Could not find step function of type: {kind}")
