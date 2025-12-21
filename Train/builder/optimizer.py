import torch


def build_optimizers(cfg, model):
    o = cfg["optim"]
    kind = o["kind"]

    if kind == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(o["lr"]), weight_decay=float(o["weight_decay"])
        )
        return optimizer
    raise ValueError(f"Could not find Optimizer of type : {kind}")
