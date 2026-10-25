"""Two ways to feed the model the same data with the same augmentation.

Baseline: torchvision ImageFolder. Every step opens 128 PNG files, decodes them with PIL
and runs the transforms one image at a time in Python.

Fast: TensorBatchLoader. Images are decoded once into a uint8 tensor (data.build_cache).
Each batch is one tensor slice, and the augmentation (random crop with padding and a
horizontal flip) runs on the whole batch at once with vectorised tensor operations.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from speedup.data import MEAN, PNG_DIR, STD


# ---------- Baseline ----------

def baseline_transforms(train: bool) -> transforms.Compose:
    steps = [transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([*steps, transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


def baseline_loader(split: str, batch_size: int, num_workers: int = 0, shuffle: bool = True) -> DataLoader:
    ds = datasets.ImageFolder(PNG_DIR / split, transform=baseline_transforms(split == "train"),
                              loader=lambda p: datasets.folder.pil_loader(p).convert("L"))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=num_workers > 0, drop_last=shuffle)


# ---------- Fast ----------

def batch_augment(x: torch.Tensor, pad: int = 2, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random crop with zero padding plus a random horizontal flip, for a whole batch.

    x: uint8 or float tensor, N x C x H x W. Each image gets its own offset and flip,
    the same augmentation as RandomCrop(28, padding=2) + RandomHorizontalFlip.
    """
    n, c, h, w = x.shape
    padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    dev = x.device
    dy = torch.randint(0, 2 * pad + 1, (n,), generator=generator).to(dev)
    dx = torch.randint(0, 2 * pad + 1, (n,), generator=generator).to(dev)
    rows = dy[:, None] + torch.arange(h, device=dev)[None, :]       # n x h
    cols = dx[:, None] + torch.arange(w, device=dev)[None, :]       # n x w
    flip = (torch.rand(n, generator=generator) < 0.5).to(dev)
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    batch_idx = torch.arange(n, device=dev)[:, None, None]
    out = padded.permute(0, 2, 3, 1)[batch_idx, rows[:, :, None], cols[:, None, :]]  # n x h x w x c
    return out.permute(0, 3, 1, 2).contiguous()


class TensorBatchLoader:
    """Yields (images, labels) batches from an in-memory uint8 tensor."""

    def __init__(self, cache_path, batch_size: int, train: bool = True, device: str = "cpu",
                 channels_last: bool = False, seed: int = 0) -> None:
        data = torch.load(cache_path)
        self.images = data["images"].to(device)
        self.labels = data["labels"].to(device)
        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.channels_last = channels_last
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        n = len(self.labels)
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.labels)
        order = torch.randperm(n, generator=self.generator) if self.train else torch.arange(n)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size].to(self.device)
            x = self.images[idx]
            if self.train:
                x = batch_augment(x, generator=self.generator)
            x = (x.float().div_(255) - MEAN) / STD
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            yield x, self.labels[idx]
