from __future__ import annotations

import torch
import torch.nn as nn

from .stem import SmallImageStem
from .blocks import BasicBlock


class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 3, base_width: int = 64):
        super().__init__()
        self.inplanes = base_width

        self.stem = SmallImageStem(in_channels=in_channels, out_channels=base_width)

        self.layer1 = self._make_layer(BasicBlock, out_ch=base_width, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, out_ch=base_width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, out_ch=base_width * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, out_ch=base_width * 8, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * getattr(BasicBlock, "expansion", 1), num_classes)

    def _make_layer(self, block: type[nn.Module], out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        """Builds one ResNet stage."""
        layers = []

        layers.append(block(self.inplanes, out_ch, stride=stride))

        expansion = getattr(block, "expansion", 1)
        self.inplanes = out_ch * expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_ch, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
