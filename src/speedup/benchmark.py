"""Measure each speed-up on its own, stacked one on top of the other.

    python -m speedup.benchmark            # timing table, chart and accuracy check

Writes benchmark.csv, benchmark.md, speedup.png and accuracy.json to results/cpu/ or
results/gpu/, depending on the device.
"""
import json
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from speedup.train import Config, benchmark, device, results_dir, train  # noqa: E402

# Each config adds one change to the one before it.
CONFIGS = [
    Config("0_baseline"),
    Config("1_dataloader_workers", num_workers=2),
    Config("2_tensor_cache_loader", loader="fast"),
    Config("3_channels_last", loader="fast", channels_last=True),
    Config("4_mixed_precision", loader="fast", channels_last=True, amp=True),
    Config("5_torch_compile", loader="fast", channels_last=True, amp=True, compile=True),
    Config("6_batch_512", loader="fast", channels_last=True, amp=True, compile=True, batch_size=512, lr=0.1),
]
FINAL = CONFIGS[5]  # the accuracy check always compares the baseline with this setup

# GPU only: CUDA graphs replay the whole step's kernels with one launch
CUDA_GRAPHS = Config("5b_cuda_graphs", loader="fast", channels_last=True, amp=True, compile=True,
                     compile_mode="reduce-overhead")


def configs_for_device() -> list[Config]:
    if device() != "cuda":
        return CONFIGS
    return CONFIGS[:6] + [CUDA_GRAPHS] + CONFIGS[6:]

LABELS = {
    "0_baseline": "Baseline: PNG files, PIL, fp32",
    "1_dataloader_workers": "+ 2 DataLoader workers",
    "2_tensor_cache_loader": "+ uint8 tensor cache, batched augmentation",
    "3_channels_last": "+ channels_last memory format",
    "4_mixed_precision": "+ mixed precision (bf16 on CPU, fp16 on GPU)",
    "5_torch_compile": "+ torch.compile",
    "5b_cuda_graphs": "+ CUDA graphs (GPU only)",
    "6_batch_512": "torch.compile + batch 512",
}


def environment() -> dict:
    return {"torch": torch.__version__, "device": device(), "threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine(), "python": platform.python_version()}


def plot(df: pd.DataFrame, path, repeats: int = 3) -> None:
    """Horizontal bar chart of throughput per config (also used to redraw from benchmark.csv)."""
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(df) + 1.2))
    colors = ["#9AA5B1" if n == CONFIGS[0].name else "#1F4E79" for n in df["name"]]
    ax.barh([LABELS[n] for n in df["name"]][::-1], df["images_per_sec"][::-1], color=colors[::-1])
    for i, (v, s) in enumerate(zip(df["images_per_sec"][::-1], df["speedup_vs_baseline"][::-1])):
        ax.text(v, i, f"  {v:,.0f} img/s ({s}x)", va="center", fontsize=9)
    ax.set_xlabel("Training throughput (images per second, higher is better)")
    ax.set_title(f"Training speed on {df['device'].iloc[0].upper()}, median of {repeats} runs")
    ax.set_xlim(0, df["images_per_sec"].max() * 1.45)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(steps: int | None = None, accuracy_epochs: int = 3, repeats: int = 3) -> pd.DataFrame:
    RESULTS = results_dir()
    # a GPU step takes ~10 ms, so it needs far more steps than the CPU for a stable timing
    steps = steps or (400 if device() == "cuda" else 60)
    rows = []
    for cfg in configs_for_device():
        # same number of images per config, so bigger batches run fewer steps
        n_steps = max(10, steps * 128 // cfg.batch_size)
        runs = [benchmark(cfg, steps=n_steps) for _ in range(repeats)]
        r = sorted(runs, key=lambda d: d["images_per_sec"])[len(runs) // 2]  # median run
        r["runs_images_per_sec"] = [d["images_per_sec"] for d in runs]
        rows.append(r)
        print(f"{cfg.name:24s} {r['images_per_sec']:9.1f} img/s   data {r['data_pct']:5.1f}%")

    df = pd.DataFrame(rows)
    base = df.loc[0, "images_per_sec"]
    df["speedup_vs_baseline"] = (df["images_per_sec"] / base).round(2)
    df["epoch_seconds_est"] = (60000 / df["images_per_sec"]).round(1)
    df.to_csv(RESULTS / "benchmark.csv", index=False)

    cols = ["name", "images_per_sec", "ms_per_step", "epoch_seconds_est", "speedup_vs_baseline",
            "data_pct", "forward_pct", "backward_pct", "optimizer_pct"]
    md = df[cols].copy()
    md.insert(1, "change", md["name"].map(LABELS))
    (RESULTS / "benchmark.md").write_text(md.drop(columns="name").to_markdown(index=False))

    plot(df, RESULTS / "speedup.png", repeats)

    # Accuracy check: the fastest setup must learn as well as the baseline
    acc = {"environment": environment()}
    for cfg in (CONFIGS[0], FINAL):
        acc[cfg.name] = train(cfg, epochs=accuracy_epochs)
        print(acc[cfg.name])
    (RESULTS / "accuracy.json").write_text(json.dumps(acc, indent=2))
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None, help="default: 60 on CPU, 400 on GPU")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    run(a.steps, a.epochs, a.repeats)
