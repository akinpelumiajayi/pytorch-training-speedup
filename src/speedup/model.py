"""A small ResNet for 28 x 28 greyscale product images (about 308,000 parameters)."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        self.layers = nn.Sequential(
            ResidualBlock(width, width),
            ResidualBlock(width, width * 2, stride=2),      # 14 x 14
            ResidualBlock(width * 2, width * 4, stride=2),  # 7 x 7
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(self.stem(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
