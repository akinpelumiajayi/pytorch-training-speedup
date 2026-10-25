"""Fashion-MNIST: 70,000 greyscale photos of Zalando catalogue products in 10 classes.

The raw files come from Zalando Research's GitHub repository. We export them to the
layout most real image projects start from, one PNG file per image in class folders,
because the cost of reading and decoding many small files is what this project fixes.

    python -m speedup.data            # download, export PNGs, build the tensor cache
"""
import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PNG_DIR = DATA_DIR / "png"
CACHE_DIR = DATA_DIR / "cache"

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASSES = ["t-shirt_top", "trouser", "pullover", "dress", "coat",
           "sandal", "shirt", "sneaker", "bag", "ankle_boot"]

# Mean and std of the training pixels, scaled to [0, 1]
MEAN, STD = 0.2860, 0.3530


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        path = RAW_DIR / name
        if not path.exists():
            print(f"Downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, path)


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    images = read_idx(RAW_DIR / FILES[f"{split}_images"])
    labels = read_idx(RAW_DIR / FILES[f"{split}_labels"])
    return images, labels


def export_png(split: str) -> None:
    """Write one PNG per image into data/png/<split>/<class>/, like a real image dataset."""
    out = PNG_DIR / split
    if out.exists() and any(out.rglob("*.png")):
        return
    images, labels = load_arrays(split)
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)
    for i, (img, y) in enumerate(zip(images, labels)):
        Image.fromarray(img, mode="L").save(out / CLASSES[y] / f"{i:05d}.png")
    print(f"Exported {len(images):,} {split} PNGs to {out}")


def build_cache(split: str) -> Path:
    """Decode every image once and store the whole split as one uint8 tensor.
    60,000 images x 28 x 28 bytes = 47 MB, small enough to keep in RAM."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{split}.pt"
    if not path.exists():
        images, labels = load_arrays(split)
        torch.save({"images": torch.from_numpy(images.copy()).unsqueeze(1),  # N x 1 x 28 x 28
                    "labels": torch.from_numpy(labels.astype(np.int64))}, path)
    return path


def prepare() -> None:
    download()
    for split in ("train", "test"):
        export_png(split)
        build_cache(split)


if __name__ == "__main__":
    prepare()
