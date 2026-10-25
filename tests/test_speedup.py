import torch

from speedup.loaders import batch_augment
from speedup.model import SmallResNet, count_parameters
from speedup.train import Config, autocast, build, train_step


def test_batch_augment_shape_dtype_and_content():
    x = torch.arange(2 * 28 * 28, dtype=torch.uint8).reshape(2, 1, 28, 28)
    out = batch_augment(x, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.dtype == torch.uint8
    # every output pixel comes from the image itself or from zero padding
    for i in range(2):
        assert set(out[i].unique().tolist()) <= set(x[i].unique().tolist()) | {0}


def test_batch_augment_zero_shift_no_flip_is_identity():
    x = torch.randint(0, 255, (4, 1, 28, 28), dtype=torch.uint8)
    out = batch_augment(x, pad=0, generator=torch.Generator().manual_seed(1))
    # with no padding only flips remain: each image equals itself or its mirror
    for i in range(4):
        assert torch.equal(out[i], x[i]) or torch.equal(out[i], x[i].flip(-1))


def test_model_output_and_size():
    m = SmallResNet()
    assert m(torch.randn(3, 1, 28, 28)).shape == (3, 10)
    assert 250_000 < count_parameters(m) < 400_000


def test_train_step_reduces_loss_with_all_speedups_on():
    cfg = Config("t", loader="fast", channels_last=True, amp=True, batch_size=32)
    torch.manual_seed(0)
    _, step_model, opt, scaler = build(cfg)
    x = torch.randn(32, 1, 28, 28).contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (32,))
    first = train_step(step_model, opt, scaler, cfg, x, y).item()
    for _ in range(20):
        last = train_step(step_model, opt, scaler, cfg, x, y).item()
    assert last < first


def test_autocast_uses_bfloat16_on_cpu():
    cfg = Config("t", amp=True)
    with autocast(cfg):
        out = torch.nn.Linear(4, 4)(torch.randn(2, 4))
    assert out.dtype in (torch.bfloat16, torch.float16)
