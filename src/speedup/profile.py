"""Profile training steps with torch.profiler to find the slowest step.

    python -m speedup.profile                 # profiles the baseline and the final config

Writes results/profile_<name>.txt (top operators) and results/trace_<name>.json
(open in chrome://tracing or https://ui.perfetto.dev).
"""
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from speedup.data import ROOT
from speedup.train import Config, autocast, build, device, make_loader, train_step

RESULTS = ROOT / "results"
STAGES = ["1_data_loading", "2_forward", "3_backward", "4_optimizer"]


def profile_config(cfg: Config, steps: int = 15, warmup: int = 5) -> dict:
    model, step_model, opt, scaler = build(cfg)
    model.train()
    it = iter(make_loader(cfg))
    for _ in range(warmup):
        x, y = next(it)
        train_step(step_model, opt, scaler, cfg, x.to(device()), y.to(device()))

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device() == "cuda" else [])
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(steps):
            with record_function("1_data_loading"):
                x, y = next(it)
                x, y = x.to(device()), y.to(device())
                if cfg.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
            with record_function("2_forward"):
                with autocast(cfg):
                    loss = torch.nn.functional.cross_entropy(step_model(x), y)
            with record_function("3_backward"):
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
            with record_function("4_optimizer"):
                scaler.step(opt)
                scaler.update()

    events = {e.key: e for e in prof.key_averages()}
    stage_ms = {s: events[s].cpu_time_total / 1000 / steps for s in STAGES if s in events}
    total = sum(stage_ms.values())

    RESULTS.mkdir(exist_ok=True)
    lines = [f"Profile: {cfg.name} ({steps} steps, batch {cfg.batch_size}, device {device()})", "",
             "Time per step by stage:"]
    lines += [f"  {s[2:]:14s} {ms:8.1f} ms  ({100 * ms / total:5.1f}%)" for s, ms in stage_ms.items()]
    lines += ["", "Top 20 operators by self CPU time:",
              prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)]
    (RESULTS / f"profile_{cfg.name}.txt").write_text("\n".join(lines))
    prof.export_chrome_trace(str(RESULTS / f"trace_{cfg.name}.json"))
    print("\n".join(lines[:8]))
    return {"name": cfg.name, **{s[2:]: round(ms, 1) for s, ms in stage_ms.items()}}


if __name__ == "__main__":
    from speedup.benchmark import CONFIGS, FINAL

    profile_config(CONFIGS[0])
    profile_config(FINAL)
