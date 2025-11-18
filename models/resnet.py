from torch import nn, Tensor
from dataclasses import dataclass
import math


@dataclass
class ResNetConfig:
    residual_block_count: int = 16
    scaling_factor: int = 2
    large_kernel_size: int = 9
    small_kernel_size: int = 3
    channels: int = 64

    def __post_init__(self):
        assert self.large_kernel_size % 2 == 1, "large_kernel_size must be odd"
        assert self.small_kernel_size % 2 == 1, "small_kernel_size must be odd"
        assert self.scaling_factor in {2, 4, 8}, (
            "Scaling factor must either be 2, 4, or 8"
        )

        self.subpixel_convolutional_block_count = int(math.log2(self.scaling_factor))


class ResidualBlock(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=config.channels,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm2d(num_features=config.channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=config.channels,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm2d(num_features=config.channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        return out


class SubpixelConvolutionalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super(SubpixelConvolutionalBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * (2**2),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=config.channels,
            kernel_size=config.large_kernel_size,
            stride=1,
            padding=config.large_kernel_size // 2,
        )
        self.prelu = nn.PReLU()

        residual_blocks = []
        for _ in range(config.residual_block_count):
            residual_blocks.append(ResidualBlock(config))
        self.residual_blocks = nn.Sequential(*residual_blocks)

        self.conv2 = nn.Conv2d(
            in_channels=config.channels,
            out_channels=config.channels,
            kernel_size=config.small_kernel_size,
            stride=1,
            padding=config.small_kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(num_features=64)

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
        # self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.prelu(out)

        residual = out
        out = self.residual_blocks(out)
        out = self.conv2(out)
        out = self.bn(out)
        out += residual

        out = self.subpixel_convolutional_blocks(out)
        out = self.conv3(out)
        # out = self.tanh(out)

        return out
