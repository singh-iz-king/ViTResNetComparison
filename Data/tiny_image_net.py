import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import Dataset
import os
from PIL import Image


class PatchifiedTinyImageNetMAE(datasets.ImageFolder):

    def __init__(self, root, patch_size=8, transform=None):
        super().__init__(root, transform=transform)
        self.patch_size = p = patch_size
        self.num_patches = (64 // p) ** 2  # Tiny ImageNet is 64x64
        self.patch_dim = p * p * 3

    def __getitem__(self, index):
        img, _ = super().__getitem__(index=index)
        c, h, w = img.shape
        p = self.patch_size

        patches = img.unfold(1, p, p).unfold(2, p, p)

        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, self.patch_dim)

        return patches


class PatchifiedTinyImageNetClassifier(datasets.ImageFolder):

    def __init__(self, root, patch_size=8, transform=None):
        super().__init__(root, transform=transform)
        self.patch_size = p = patch_size
        self.num_patches = (64 // p) ** 2  # Tiny ImageNet is 64x64
        self.patch_dim = p * p * 3

    def __getitem__(self, index):
        img, y = super().__getitem__(index=index)
        p = self.patch_size

        patches = img.unfold(1, p, p).unfold(2, p, p)

        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, self.patch_dim)

        return patches, y


class PatchifiedTinyImageNetSimCLR(datasets.ImageFolder):
    def __init__(self, root, patch_size=8, transform=None):
        super().__init__(
            root, transform=None
        )  # Transformation will be applied manually
        self.p = patch_size
        self.patch_dim = 3 * patch_size * patch_size
        self.transform = transform

    def patchify(self, img):
        patches = img.unfold(1, self.p, self.p).unfold(2, self.p, self.p)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, self.patch_dim)
        return patches

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)  # PIL
        x1 = self.transform(img)  # tensor (3,64,64)
        x2 = self.transform(img)  # independent aug
        return self.patchify(x1), self.patchify(x2)


def read_val_annotations(path):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                filename, wnid = parts[0], parts[1]
                mapping[filename] = wnid
    return mapping


class PatchifiedTinyImageNetClassifierVal(Dataset):

    def __init__(self, val_root, class_to_idx, patch_size=8, transform=None):
        super().__init__()
        self.val_root = val_root
        self.images_dir = os.path.join(val_root, "images")
        self.ann_path = os.path.join(val_root, "val_annotations.txt")
        self.transform = transform

        self.filename_to_wnid = read_val_annotations(self.ann_path)
        self.filenames = sorted(self.filename_to_wnid.keys())

        self.class_to_idx = class_to_idx
        self.patch_size = p = patch_size
        self.num_patches = (64 // p) ** 2  # Tiny ImageNet is 64x64
        self.patch_dim = p * p * 3

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fn = self.filenames[index]
        wnid = self.filename_to_wnid[fn]
        y = self.class_to_idx[wnid]

        path = os.path.join(self.images_dir, fn)
        img = Image.open(path).convert("RGB")  # PIL

        if self.transform:
            img = self.transform(img)  # Tensor (3,64,64)

        c, h, w = img.shape
        p = self.patch_size

        patches = img.unfold(1, p, p).unfold(2, p, p)

        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, self.patch_dim)

        return patches, y


mae_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop((64, 64), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

simclr_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop((64, 64), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

vit_train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(
            (64, 64),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

vit_eval_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((64, 64), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
