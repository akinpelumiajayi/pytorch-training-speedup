# Profiling and Speeding Up PyTorch Training: 3.5x Faster, Same Accuracy

I profiled a PyTorch image classifier training on real Zalando product photos, found where the time went, and cut a 3-epoch training run from **452 seconds to 130 seconds** (3.5x faster). Test accuracy stayed level: **90.6% before, 90.4% after**.

Every change is measured on its own, so you can see which ones paid off and which did not.

![Training throughput for each change](results/speedup.png)

## The task

- **Data:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), 70,000 greyscale photos of Zalando catalogue products (t-shirts, trousers, bags, sneakers and 6 more classes). The raw files are exported as one PNG per image in class folders, the layout most real image projects start from.
- **Model:** a small ResNet (3 residual blocks, 308,000 parameters) trained with SGD, Nesterov momentum and a one-cycle learning rate.
- **Augmentation:** random crop with 2-pixel padding and a random horizontal flip.
- **Hardware:** 2-core Intel Xeon CPU with AMX and AVX-512 bfloat16 support, PyTorch 2.14. No GPU. The code switches to float16 mixed precision automatically on a CUDA GPU.

## Step 1: profile the baseline

The baseline is the standard tutorial setup: `torchvision.datasets.ImageFolder` reading PNG files, PIL transforms one image at a time, float32 maths, batch size 128.

`python -m speedup.profile` wraps each stage of the training step in `torch.profiler.record_function`:

| Stage | Time per step | Share |
|---|---:|---:|
| Data loading | 45.4 ms | 13.4% |
| Forward pass | 101.6 ms | 29.9% |
| **Backward pass** | **190.9 ms** | **56.3%** |
| Optimizer step | 1.5 ms | 0.4% |

The slowest step is the backward pass. The top operator is `aten::convolution_backward` at **40% of all CPU time**, with the forward convolution (`aten::mkldnn_convolution`) next at 18%. Data loading takes 13%, enough to matter but not the main cost. So the biggest win had to come from making the convolution maths cheaper, with a faster data pipeline second.

Full operator tables: [`results/profile_0_baseline.txt`](results/profile_0_baseline.txt) and [`results/profile_5_torch_compile.txt`](results/profile_5_torch_compile.txt).

## Step 2: apply one change at a time

Each row adds one change on top of the row above. Throughput is the median of 3 runs.

| Change | Images/sec | ms/step | Speed-up | Data share |
|---|---:|---:|---:|---:|
| Baseline: PNG files, PIL, float32 | 392 | 326 | 1.00x | 11.0% |
| + 2 DataLoader workers | 400 | 320 | 1.02x | 0.9% |
| + uint8 tensor cache, batched augmentation | 489 | 262 | 1.25x | 0.5% |
| + channels_last memory format | 500 | 256 | 1.27x | 0.5% |
| **+ bfloat16 mixed precision** | **1,210** | **106** | **3.09x** | 1.1% |
| **+ torch.compile** | **1,512** | **85** | **3.86x** | 1.4% |
| + batch 512 (slower, not kept) | 1,083 | 473 | 2.76x | 2.2% |

Full table with forward, backward and optimizer shares: [`results/benchmark.md`](results/benchmark.md).

### What each change does, and why it helped or did not

**DataLoader workers (1.02x).** Two worker processes hid the data loading time (11% down to 0.9%), but the machine has only 2 cores. The workers took CPU away from the convolutions, so the overall gain was close to zero.

**Custom tensor-cache loader (1.25x).** [`loaders.py`](src/speedup/loaders.py) decodes all 60,000 PNGs once into a single 47 MB `uint8` tensor. Each batch becomes one tensor slice, and the crop and flip run on the whole batch with vectorised indexing in [`batch_augment`](src/speedup/loaders.py) instead of 128 separate PIL calls. Data loading fell to 0.5% of step time and freed both cores for the model.

**channels_last (1.02x).** Stores images as NHWC, the layout oneDNN convolution kernels prefer. A small gain on its own, and it sets up the next step.

**bfloat16 mixed precision (2.4x on top).** The biggest win, aimed straight at the convolution bottleneck. `torch.autocast` runs convolutions in bfloat16, which this CPU's AMX units process far faster than float32. bfloat16 keeps float32's exponent range, so no gradient scaling is needed. On a CUDA GPU the same code uses float16 with `GradScaler`.

**torch.compile (1.25x on top).** Fuses the batch norm, ReLU and residual additions into fewer kernels, cutting memory traffic between operations. The first steps pay a one-off compile cost (under a minute here), which is why the benchmark warms up before timing.

**Batch 512 (slower, not kept).** Larger batches often help on a GPU. On a 2-core CPU, the bigger activations no longer fit in cache and throughput dropped by 28%. The final setup keeps batch 128.

## Step 3: check accuracy did not suffer

Speed means nothing if the model learns worse. Both setups trained for 3 full epochs from the same seed (`results/accuracy.json`):

| Setup | Seconds per epoch | 3-epoch training time | Test accuracy |
|---|---:|---:|---:|
| Baseline | 150.8 | 452 s | 90.63% |
| Optimised | 43.4 | 130 s | 90.40% |

The 0.23-point gap is small. Each setup ran once, so expect the exact figures to move a little between runs.

## Run

```bash
git clone https://github.com/akinpelumiajayi/pytorch-training-speedup.git
cd pytorch-training-speedup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src

python -m speedup.data         # download Fashion-MNIST, export 70,000 PNGs, build the tensor cache
python -m speedup.profile      # profile the baseline and the optimised setup
python -m speedup.benchmark    # timing table, chart and accuracy check (about 15 minutes on 2 CPU cores)
pytest -q                      # 5 tests
```

The profiler also writes Chrome traces to `results/trace_*.json`. Open them in [Perfetto](https://ui.perfetto.dev) to see every operator on a timeline.

## Note

Your numbers will differ with your hardware. A CPU without bfloat16 support gains less from mixed precision. A GPU shifts the bottleneck towards data loading, where the tensor-cache loader matters more.

## Project structure

```
src/speedup/
├── data.py        download, PNG export, uint8 tensor cache
├── loaders.py     baseline ImageFolder loader and the fast TensorBatchLoader
├── model.py       small ResNet
├── train.py       training step with switchable speed-ups, timed benchmark, full training
├── profile.py     torch.profiler stage breakdown and operator tables
└── benchmark.py   ablation table, chart and accuracy check
results/           measured outputs from the run above
tests/             pytest suite, run by GitHub Actions on every push
```

## Author

Akinpelumi Michael Ajayi, MSc Artificial Intelligence (Distinction)
[LinkedIn](https://www.linkedin.com/in/akinpelumi-michael-ajayi-) | [GitHub](https://github.com/akinpelumiajayi)

## Licence

MIT. Fashion-MNIST is released by Zalando Research under the MIT licence.
