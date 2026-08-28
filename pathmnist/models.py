from __future__ import annotations

import torch
from torch import Tensor
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )


class MultiscaleBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        branch_channels = max(4, outputs // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(inputs, branch_channels, kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for kernel in (1, 3, 5)
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv2d(branch_channels * 3, outputs, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        branches = [branch(inputs) for branch in self.branches]
        return self.projection(torch.cat(branches, dim=1))


class SmallResNet(nn.Module):
    def __init__(self, classes: int, multiscale: bool = False) -> None:
        super().__init__()
        self.multiscale = multiscale
        width = [32, 64, 128, 256]
        self.stem = ConvBlock(3, width[0])
        self.layer1 = nn.Sequential(ConvBlock(width[0], width[1], stride=2), ConvBlock(width[1], width[1]))
        self.layer2 = nn.Sequential(ConvBlock(width[1], width[2], stride=2), ConvBlock(width[2], width[2]))
        self.layer3 = nn.Sequential(ConvBlock(width[2], width[3], stride=2), ConvBlock(width[3], width[3]))
        if multiscale:
            self.fusion1 = MultiscaleBlock(width[1], width[1])
            self.fusion2 = MultiscaleBlock(width[2], width[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width[3], classes)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = self.stem(inputs)
        inputs = self.layer1(inputs)
        if self.multiscale:
            inputs = self.fusion1(inputs)
        inputs = self.layer2(inputs)
        if self.multiscale:
            inputs = self.fusion2(inputs)
        inputs = self.layer3(inputs)
        return self.classifier(self.pool(inputs).flatten(1))
