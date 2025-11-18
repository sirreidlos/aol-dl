from dataclasses import dataclass
import torch
from torch import nn

from .resnet import ResNet, ResNetConfig

GeneratorConfig = ResNetConfig


@dataclass
class DiscriminatorConfig:
    convolutional_block_count: int = 7
    kernel_size: int = 3
    channels: int = 64
    dense_size: int = 1024


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(ConvolutionalBlock, self).__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(num_features=out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out


class Generator(nn.Module):
    def __init__(self, config: GeneratorConfig) -> None:
        super(Generator, self).__init__()
        self.resnet = ResNet(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig) -> None:
        super(Discriminator, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=config.channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.lrelu = nn.LeakyReLU(0.2)

        convolutional_blocks = []

        channels = config.channels
        for i in range(0, config.convolutional_block_count):
            if i % 2 == 0:
                convolutional_blocks.append(
                    ConvolutionalBlock(
                        in_channels=channels,
                        out_channels=channels,
                        stride=2,
                    )
                )
            else:
                convolutional_blocks.append(
                    ConvolutionalBlock(
                        in_channels=channels,
                        out_channels=channels * 2,
                        stride=1,
                    )
                )
                channels *= 2

        self.convolutional_blocks = nn.Sequential(*convolutional_blocks)

        mlp = [
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(channels * 6 * 6, config.dense_size),
            nn.LeakyReLU(0.2),
            nn.Linear(config.dense_size, 1),
        ]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        raw logits
        """
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.convolutional_blocks(out)
        out = self.mlp(out)

        return out
