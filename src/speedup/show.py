"""Preview Fashion-MNIST images in the terminal.

Each character cell shows two pixels: the upper half block (▀) is coloured with the top
pixel and its background with the bottom one, using 24-bit ANSI colour codes.

    python -m speedup.show                         # first 8 training images
    python -m speedup.show --split test -n 12 --random
    python -m speedup.show --class sneaker --augment
"""
import argparse
import sys

import torch

from speedup.data import CLASSES, build_cache, download
from speedup.loaders import batch_augment

PER_ROW = 4
GAP = "  "


def render(img: torch.Tensor) -> list[str]:
    """28 x 28 (or 1 x 28 x 28) uint8 image -> 14 lines of coloured half blocks."""
    img = img.squeeze(0)
    h, w = img.shape
    lines = []
    for r in range(0, h, 2):
        line = ""
        for c in range(w):
            top = int(img[r, c])
            bot = int(img[r + 1, c]) if r + 1 < h else 0
            line += f"\x1b[38;2;{top};{top};{top}m\x1b[48;2;{bot};{bot};{bot}m▀"
        lines.append(line + "\x1b[0m")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Show dataset images in the terminal.")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("-n", type=int, default=8, help="number of images")
    p.add_argument("--start", type=int, default=0, help="first index (ignored with --random)")
    p.add_argument("--random", action="store_true", help="pick random images")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--class", dest="cls", choices=CLASSES, help="only show this class")
    p.add_argument("--augment", action="store_true", help="apply the training augmentation")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    download()
    data = torch.load(build_cache(args.split))
    images, labels = data["images"], data["labels"]

    pool = torch.arange(len(labels))
    if args.cls:
        pool = pool[labels == CLASSES.index(args.cls)]
    if args.random:
        g = torch.Generator()
        g.manual_seed(args.seed) if args.seed is not None else g.seed()
        idx = pool[torch.randperm(len(pool), generator=g)[:args.n]]
    else:
        idx = pool[args.start:args.start + args.n]

    batch = images[idx]
    if args.augment:
        batch = batch_augment(batch)

    width = batch.shape[-1]
    for s in range(0, len(idx), PER_ROW):
        chunk = range(s, min(s + PER_ROW, len(idx)))
        print(GAP.join(f"#{int(idx[i])} {CLASSES[labels[idx[i]]]}"[:width].ljust(width) for i in chunk))
        blocks = [render(batch[i]) for i in chunk]
        for row in zip(*blocks):
            print(GAP.join(row))
        print()


if __name__ == "__main__":
    main()
