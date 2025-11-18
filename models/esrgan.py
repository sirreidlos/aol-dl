from dataclasses import dataclass
import torch
from torch import nn
import math

from models.resnet import SubpixelConvolutionalBlock


@dataclass
class ESRGANGeneratorConfig:
    rrdb_block_count: int = 16
    scaling_factor: int = 2
    large_kernel_size: int = 9
    small_kernel_size: int = 3
    channels: int = 64
    beta: float = 0.2

    def __post_init__(self):
        assert self.large_kernel_size % 2 == 1, "large_kernel_size must be odd"
        assert self.small_kernel_size % 2 == 1, "small_kernel_size must be odd"
        assert self.scaling_factor in {2, 4, 8}, (
            "Scaling factor must either be 2, 4, or 8"
        )

        self.subpixel_convolutional_block_count = int(math.log2(self.scaling_factor))


class RDB(nn.Module):
    def __init__(self, config: ESRGANGeneratorConfig) -> None:
        super().__init__()

        self.beta = config.beta

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels=config.channels,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.conv2 = nn.Conv2d(
            in_channels=config.channels * 2,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.conv3 = nn.Conv2d(
            in_channels=config.channels * 3,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.conv4 = nn.Conv2d(
            in_channels=config.channels * 4,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.conv5 = nn.Conv2d(
            in_channels=config.channels * 5,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.beta + x


class RRDB(nn.Module):
    def __init__(self, config: ESRGANGeneratorConfig) -> None:
        super().__init__()
        self.rdb1 = RDB(config)
        self.rdb2 = RDB(config)
        self.rdb3 = RDB(config)

        self.beta = config.beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)

        return out * self.beta + x


class ESRGANGenerator(nn.Module):
    def __init__(self, config: ESRGANGeneratorConfig) -> None:
        super(ESRGANGenerator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=config.channels,
            kernel_size=config.large_kernel_size,
            stride=1,
            padding=config.large_kernel_size // 2,
        )
        self.prelu = nn.PReLU()

        residual_blocks = []
        for _ in range(config.rrdb_block_count):
            residual_blocks.append(RRDB(config))
        self.rrdb_blocks = nn.Sequential(*residual_blocks)

        self.conv2 = nn.Conv2d(
            in_channels=config.channels,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )

        subpixel_convolutional_blocks = []
        for _ in range(config.subpixel_convolutional_block_count):
            subpixel_convolutional_blocks.append(
                SubpixelConvolutionalBlock(config.channels, config.small_kernel_size)
            )
        self.subpixel_convolutional_blocks = nn.Sequential(
            *subpixel_convolutional_blocks
        )

        self.conv3 = nn.Conv2d(
            in_channels=config.channels,
            out_channels=3,
            kernel_size=config.large_kernel_size,
            stride=1,
            padding=config.large_kernel_size // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.prelu(out)

        residual = out
        out = self.rrdb_blocks(out)
        out = self.conv2(out)
        out += residual

        out = self.subpixel_convolutional_blocks(out)
        out = self.conv3(out)

        return out


class ESRGANDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(ESRGANDiscriminator, self).__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
