"""Training loop with switchable speed-ups, plus a timed benchmark of each step.

Every config trains the same model on the same data with the same augmentation.
Only the way data is loaded and the way the maths runs change.
"""
import contextlib
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from speedup.data import CACHE_DIR, ROOT
from speedup.loaders import TensorBatchLoader, baseline_loader
from speedup.model import SmallResNet


@dataclass
class Config:
    name: str
    loader: str = "baseline"       # "baseline" (PNG files + PIL) or "fast" (uint8 tensor cache)
    num_workers: int = 0           # DataLoader worker processes, baseline loader only
    batch_size: int = 128
    channels_last: bool = False    # NHWC memory layout for convolutions
    amp: bool = False              # mixed precision: bfloat16 on CPU, float16 on CUDA
    compile: bool = False          # torch.compile the model
    compile_mode: str = "default"  # "reduce-overhead" adds CUDA graphs (GPU only)
    lr: float = 0.05


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def results_dir():
    """results/cpu or results/gpu, so runs on different hardware never overwrite each other."""
    path = ROOT / "results" / ("gpu" if device() == "cuda" else "cpu")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_loader(cfg: Config, split: str = "train"):
    if cfg.loader == "baseline":
        return baseline_loader(split, cfg.batch_size, cfg.num_workers, shuffle=split == "train")
    return TensorBatchLoader(CACHE_DIR / f"{split}.pt", cfg.batch_size, train=split == "train",
                             device=device(), channels_last=cfg.channels_last)


def autocast(cfg: Config):
    if not cfg.amp:
        return contextlib.nullcontext()
    dtype = torch.float16 if device() == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device(), dtype=dtype)


def build(cfg: Config, seed: int = 0):
    torch.manual_seed(seed)
    model = SmallResNet().to(device())
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # float16 needs loss scaling; bfloat16 has the float32 range and does not
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device() == "cuda")
    step_model = torch.compile(model, mode=cfg.compile_mode) if cfg.compile else model
    return model, step_model, opt, scaler


def train_step(step_model, opt, scaler, cfg, x, y, timings=None):
    """One optimisation step. If timings is a dict, adds forward/backward/optimizer seconds."""
    t0 = time.perf_counter()
    with autocast(cfg):
        loss = nn.functional.cross_entropy(step_model(x), y)
    if timings is not None:
        sync(); t1 = time.perf_counter(); timings["forward"] += t1 - t0
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if timings is not None:
        sync(); t2 = time.perf_counter(); timings["backward"] += t2 - t1
    scaler.step(opt)
    scaler.update()
    if timings is not None:
        sync(); timings["optimizer"] += time.perf_counter() - t2
    return loss


def benchmark(cfg: Config, steps: int = 60, warmup: int = 10) -> dict:
    """Time `steps` training steps after `warmup` untimed steps. Returns throughput and a
    breakdown of where the time went."""
    model, step_model, opt, scaler = build(cfg)
    model.train()
    loader = make_loader(cfg)
    it = iter(loader)
    timings = {"data": 0.0, "forward": 0.0, "backward": 0.0, "optimizer": 0.0}

    def next_batch():
        nonlocal it
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device(), non_blocking=True), y.to(device(), non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x, y

    for _ in range(warmup):
        x, y = next_batch()
        train_step(step_model, opt, scaler, cfg, x, y)
    sync()

    start = time.perf_counter()
    for _ in range(steps):
        t = time.perf_counter()
        x, y = next_batch()
        sync()
        timings["data"] += time.perf_counter() - t
        train_step(step_model, opt, scaler, cfg, x, y, timings)
    sync()
    total = time.perf_counter() - start

    images = steps * cfg.batch_size
    return {
        **asdict(cfg),
        "device": device(),
        "steps": steps,
        "seconds": round(total, 3),
        "images_per_sec": round(images / total, 1),
        "ms_per_step": round(1000 * total / steps, 2),
        **{f"{k}_pct": round(100 * v / total, 1) for k, v in timings.items()},
    }


@torch.no_grad()
def evaluate(model, cfg: Config) -> float:
    model.eval()
    correct = total = 0
    for x, y in make_loader(cfg, "test"):
        x, y = x.to(device()), y.to(device())
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with autocast(cfg):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def train(cfg: Config, epochs: int = 3, seed: int = 0, on_step=None, on_epoch=None,
          eval_each_epoch: bool = False, log_every: int = 20) -> dict:
    """Full training run: returns wall-clock time per epoch and test accuracy.

    Optional hooks for a live display (python -m speedup.train). They are off by default,
    so benchmark timings are unaffected:
      on_step(epoch, step, steps_per_epoch, mean_loss, lr)   every `log_every` steps
      on_epoch(epoch, epoch_seconds, test_accuracy, model)    after each epoch
    Loss is summed on the device and read back only every `log_every` steps, because each
    read forces the CPU to wait for the GPU. Evaluation is never counted in epoch time.
    """
    model, step_model, opt, scaler = build(cfg, seed)
    loader = make_loader(cfg)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr * 2, total_steps=epochs * len(loader))
    epoch_times, accuracy = [], None
    for epoch in range(epochs):
        model.train()
        running, count = torch.zeros((), device=device()), 0
        sync(); t = time.perf_counter()
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device(), non_blocking=True), y.to(device(), non_blocking=True)
            if cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            loss = train_step(step_model, opt, scaler, cfg, x, y)
            sched.step()
            if on_step is not None:
                running += loss.detach().float()
                count += 1
                if i % log_every == 0 or i == len(loader):
                    on_step(epoch, i, len(loader), (running / count).item(), sched.get_last_lr()[0])
                    running.zero_()
                    count = 0
        sync(); epoch_times.append(time.perf_counter() - t)
        if eval_each_epoch or epoch == epochs - 1:
            accuracy = evaluate(model, cfg)
        if on_epoch is not None:
            on_epoch(epoch, epoch_times[-1], accuracy if eval_each_epoch or epoch == epochs - 1 else None, model)
    return {"name": cfg.name, "epochs": epochs, "epoch_seconds": [round(s, 1) for s in epoch_times],
            "train_seconds": round(sum(epoch_times), 1), "test_accuracy": round(accuracy, 4)}


if __name__ == "__main__":
    from speedup.live import main

    main()
