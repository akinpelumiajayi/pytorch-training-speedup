"""Live training display for `python -m speedup.train`.

    python -m speedup.train                    # optimised setup, 3 epochs, live dashboard
    python -m speedup.train --compare          # baseline then optimised, side by side
    python -m speedup.train --config 0_baseline --epochs 5
    python -m speedup.train --list             # show every config name

While training, the terminal shows the hardware, the config, a progress bar with the
live loss, learning rate and images/sec, and a table that gains a row per epoch with
the test accuracy. Every run is saved to results/<cpu|gpu>/runs/<time>_<name>/ as
evidence: the terminal transcript (train_log.svg / .html), per-step and per-epoch CSVs,
a summary.json, training curves (curves.png) and the trained weights (model.pt).
"""
import argparse
import json
import platform
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rich import box  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, \
    TimeRemainingColumn  # noqa: E402
from rich.table import Table  # noqa: E402

from speedup.benchmark import CONFIGS, FINAL, LABELS, configs_for_device  # noqa: E402
from speedup.model import SmallResNet, count_parameters  # noqa: E402
from speedup.train import Config, device, results_dir, train  # noqa: E402

TRAIN_IMAGES = 60_000


def hardware() -> str:
    if device() == "cuda":
        props = torch.cuda.get_device_properties(0)
        return f"{props.name} ({props.total_memory / 2**30:.0f} GB, CUDA {torch.version.cuda})"
    return f"CPU: {platform.processor() or platform.machine()} ({torch.get_num_threads()} threads)"


def header(console: Console, configs: list[Config], epochs: int, seed: int) -> None:
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan")
    info.add_column()
    info.add_row("Hardware", hardware())
    info.add_row("Software", f"PyTorch {torch.__version__}, Python {platform.python_version()}")
    info.add_row("Data", "Fashion-MNIST: 60,000 train / 10,000 test product photos, 10 classes")
    info.add_row("Model", f"Small ResNet, {count_parameters(SmallResNet()):,} parameters")
    info.add_row("Run", f"{epochs} epochs, seed {seed}, started {datetime.now():%Y-%m-%d %H:%M:%S}")
    for cfg in configs:
        flags = [f"loader={cfg.loader}", f"batch={cfg.batch_size}"]
        flags += [f for f, on in [("workers=%d" % cfg.num_workers, cfg.num_workers), ("channels_last", cfg.channels_last),
                                  ("mixed precision", cfg.amp), (f"torch.compile[{cfg.compile_mode}]", cfg.compile)] if on]
        info.add_row(f"Config {cfg.name}", ", ".join(flags))
    console.print(Panel(info, title="[bold]PyTorch training speed-up: live training run", border_style="cyan"))


def run_one(console: Console, cfg: Config, epochs: int, seed: int, log_every: int) -> dict:
    steps_log, epochs_log = [], []
    epoch_table = Table(box=box.SIMPLE_HEAVY, title=f"[bold]{cfg.name}[/]  {LABELS.get(cfg.name, '')}")
    for col, just in [("Epoch", "right"), ("Train loss", "right"), ("Test accuracy", "right"),
                      ("Epoch time", "right"), ("Images/sec", "right")]:
        epoch_table.add_column(col, justify=just)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"), BarColumn(bar_width=None), MofNCompleteColumn(),
        TextColumn("loss [yellow]{task.fields[loss]}"), TextColumn("lr {task.fields[lr]}"),
        TextColumn("[green]{task.fields[speed]} img/s"), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console, expand=True)
    warmup = " (compiling the model, first steps are slow)" if cfg.compile else ""
    task = progress.add_task(f"Epoch 1/{epochs}{warmup}", total=None, loss="-", lr="-", speed="-")
    state = {"t": time.perf_counter(), "step": 0, "loss": float("nan")}

    def on_step(epoch, step, steps_per_epoch, loss, lr):
        now = time.perf_counter()
        speed = (step - state["step"]) * cfg.batch_size / (now - state["t"])
        state.update(t=now, step=step, loss=loss)
        global_step = epoch * steps_per_epoch + step
        steps_log.append({"epoch": epoch + 1, "step": global_step, "loss": loss, "lr": lr, "images_per_sec": speed})
        progress.update(task, description=f"Epoch {epoch + 1}/{epochs}", total=steps_per_epoch, completed=step,
                        loss=f"{loss:.4f}", lr=f"{lr:.4f}", speed=f"{speed:,.0f}")

    def on_epoch(epoch, seconds, accuracy, model):
        acc = f"{100 * accuracy:.2f}%" if accuracy is not None else "-"
        epoch_table.add_row(str(epoch + 1), f"{state['loss']:.4f}", acc, f"{seconds:.1f} s",
                            f"{TRAIN_IMAGES / seconds:,.0f}")
        epochs_log.append({"epoch": epoch + 1, "train_loss": state["loss"], "test_accuracy": accuracy,
                           "epoch_seconds": round(seconds, 2)})
        state.update(t=time.perf_counter(), step=0)  # evaluation time is not training speed
        if epoch == epochs - 1:
            state["model"] = model
        else:
            progress.update(task, description=f"Epoch {epoch + 2}/{epochs}", completed=0)

    with Live(Group(epoch_table, progress), console=console, refresh_per_second=8):
        result = train(cfg, epochs=epochs, seed=seed, on_step=on_step, on_epoch=on_epoch,
                       eval_each_epoch=True, log_every=log_every)
    result["images_per_sec"] = round(epochs * TRAIN_IMAGES / result["train_seconds"], 1)
    return {"result": result, "steps": steps_log, "epochs": epochs_log, "model": state.pop("model")}


def steady_epoch_seconds(result: dict) -> float:
    """Mean epoch time after the first epoch. Epoch 1 also pays one-off costs: torch.compile
    compiling the model, and the OS reading the PNG files from disk for the first time."""
    times = result["epoch_seconds"]
    return sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]


def summary(console: Console, runs: list[dict]) -> None:
    compare = len(runs) > 1
    table = Table(title="[bold]Result", box=box.ROUNDED)
    cols = ["Config", "Total time", "Steady epoch", "Steady img/s", "Test accuracy"]
    cols += ["Speed-up (total)", "Speed-up (steady)"] if compare else []
    for col in cols:
        table.add_column(col, justify="left" if col == "Config" else "right")
    base = runs[0]["result"]
    for run in runs:
        r = run["result"]
        steady = steady_epoch_seconds(r)
        r["steady_epoch_seconds"] = round(steady, 2)
        r["steady_images_per_sec"] = round(TRAIN_IMAGES / steady, 1)
        row = [r["name"], f"{r['train_seconds']:.1f} s", f"{steady:.1f} s", f"{TRAIN_IMAGES / steady:,.0f}",
               f"{100 * r['test_accuracy']:.2f}%"]
        if compare:
            row += [f"{base['train_seconds'] / r['train_seconds']:.2f}x",
                    f"[bold green]{steady_epoch_seconds(base) / steady:.2f}x"]
        table.add_row(*row)
    console.print()
    console.print(table)
    if runs[0]["result"]["epochs"] > 1:
        console.print("[dim]Total time includes epoch 1, which pays one-off costs: compiling the model with "
                      "torch.compile and the first read of the PNG files from disk. Steady = epochs 2 onwards.")


def save(console: Console, runs: list[dict], args) -> None:
    name = "compare" if args.compare else runs[0]["result"]["name"]
    out = results_dir() / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    out.mkdir(parents=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run in runs:
        n = run["result"]["name"]
        pd.DataFrame(run["steps"]).to_csv(out / f"steps_{n}.csv", index=False)
        pd.DataFrame(run["epochs"]).to_csv(out / f"epochs_{n}.csv", index=False)
        torch.save(run["model"].state_dict(), out / f"model_{n}.pt")
        steps = pd.DataFrame(run["steps"])
        ax1.plot(steps["step"], steps["loss"], label=n)
        epochs = pd.DataFrame(run["epochs"])
        ax2.plot(epochs["epoch_seconds"].cumsum(), 100 * epochs["test_accuracy"], marker="o", label=n)
    ax1.set(xlabel="Training step", ylabel="Train loss", title="Training loss")
    ax2.set(xlabel="Training time (seconds)", ylabel="Test accuracy (%)", title="Test accuracy over training time")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Fashion-MNIST ResNet on {hardware()}")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps({
        "hardware": hardware(), "torch": torch.__version__, "python": platform.python_version(),
        "epochs": args.epochs, "seed": args.seed, "finished": datetime.now().isoformat(timespec="seconds"),
        "runs": [run["result"] for run in runs]}, indent=2))
    console.print(f"\n[bold green]Saved evidence to[/] {out}")
    console.print("  train_log.svg / train_log.html  this terminal session")
    console.print("  curves.png                      loss and accuracy curves")
    console.print("  summary.json, steps_*.csv, epochs_*.csv, model_*.pt")
    console.save_svg(str(out / "train_log.svg"), title="python -m speedup.train", clear=False)
    console.save_html(str(out / "train_log.html"))


def main() -> None:
    configs = {c.name: c for c in configs_for_device()}
    p = argparse.ArgumentParser(prog="python -m speedup.train", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=FINAL.name, choices=list(configs), help=f"default: {FINAL.name}")
    p.add_argument("--compare", action="store_true", help=f"train {CONFIGS[0].name} then {FINAL.name} and compare")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20, help="refresh the loss every N steps")
    p.add_argument("--list", action="store_true", help="list the configs and exit")
    args = p.parse_args()

    console = Console(record=True)
    if args.list:
        for name, cfg in configs.items():
            console.print(f"[bold]{name:22s}[/] {LABELS.get(name, '')}")
        return
    chosen = [CONFIGS[0], FINAL] if args.compare else [configs[args.config]]
    header(console, chosen, args.epochs, args.seed)
    runs = [run_one(console, cfg, args.epochs, args.seed, args.log_every) for cfg in chosen]
    summary(console, runs)
    save(console, runs, args)


if __name__ == "__main__":
    main()
