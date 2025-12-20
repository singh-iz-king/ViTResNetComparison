import torch
from torch.utils.data import DataLoader

from Data.tiny_image_net import (
    PatchifiedTinyImageNetClassifier,
    PatchifiedTinyImageNetMAE,
    PatchifiedTinyImageNetSimCLR,
    PatchifiedTinyImageNetClassifierVal,
    mae_transforms,
    simclr_transforms,
)


def build_loaders(cfg):
    d = cfg["data"]
    kind = d["kind"]

    if kind == "tinyimagenet_mae":
        ds = PatchifiedTinyImageNetMAE(
            d["root"], d["patch_size"], transform=mae_transforms
        )
        train_loader = DataLoader(ds, batch_size=d["batch_size"], shuffle=True)

        return train_loader, None

    if kind == "tinyimagenet_simclr":
        ds = PatchifiedTinyImageNetSimCLR(
            d["root"], d["patch_size"], transform=simclr_transforms
        )
        train_loader = DataLoader(ds, batch_size=d["batch_size"], shuffle=True)

        return train_loader, None

    if kind == "tinyimagenet":
        ds_train = PatchifiedTinyImageNetClassifier(
            d["root"], d["patch_size"], transform=None
        )
        ds_val = PatchifiedTinyImageNetClassifierVal(
            d["root"], ds_train.class_to_idx, d["patch_size"], transform=None
        )
        train_loader, val_loader = DataLoader(
            ds_train, d["batch_size"], shuffle=True
        ), DataLoader(ds_val, d["batch_size"], shuffle=False)
        return train_loader, val_loader
    raise ValueError(f"Uknown data kind: {kind}")
