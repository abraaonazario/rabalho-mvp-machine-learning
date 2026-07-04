"""
models.py — Arquiteturas CNN v2.0
===================================
Implementa:
  - RF-08: SmallResNet aprimorada com num_blocks por camada (deeper)
  - Backward compatibility: num_blocks=1 reproduz comportamento da v1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """Arquitetura CNN canônica de baseline.

    Duas camadas convolucionais seguidas de MaxPooling e classificador
    fully-connected. Serve como limite inferior de desempenho.

    Args:
        num_classes: Número de classes de saída.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        # 64x64 -> conv1(3x3) -> 62x62 -> pool -> 31x31
        # 31x31 -> conv2(3x3) -> 29x29 -> pool -> 14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagação forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallVGG(nn.Module):
    """Arquitetura inspirada na família VGG com BatchNorm e Dropout.

    Três blocos de dupla convolução 3×3 com MaxPooling, seguidos de
    classificador fully-connected com Dropout de 30%.

    Args:
        num_classes: Número de classes de saída.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Bloco 1: 3 → 32
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)

        # Bloco 2: 32 → 64
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        # Bloco 3: 64 → 128
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)

        # Classificador: 64x64 → pool(32) → pool(16) → pool(8) → 128×8×8
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagação forward."""
        x = self.pool(F.relu(self.bn1_2(self.conv1_2(F.relu(self.bn1_1(self.conv1_1(x)))))))
        x = self.pool(F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool(F.relu(self.bn3_2(self.conv3_2(F.relu(self.bn3_1(self.conv3_1(x)))))))
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Bloco residual básico com shortcut connection.

    Implementa a identidade ou projeção via convolução 1×1 quando há
    mudança de dimensão (stride > 1 ou canais diferentes).

    Args:
        in_channels: Número de canais de entrada.
        out_channels: Número de canais de saída.
        stride: Stride da primeira convolução (>1 aplica downsampling).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagação forward com skip connection."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    """Arquitetura ResNet pequena com múltiplos blocos residuais por camada (v2.0).

    Versão aprimorada (RF-08) que suporta ``num_blocks`` blocos residuais por
    estágio, aumentando a profundidade e capacidade representacional.
    Com ``num_blocks=1`` o comportamento é idêntico à v1.0 (backward compatibility).

    Args:
        num_classes: Número de classes de saída.
        num_blocks: Número de ResidualBlocks por estágio (default: 2 para v2.0).
    """

    def __init__(self, num_classes: int = 10, num_blocks: int = 2) -> None:
        super().__init__()
        self._in_channels = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # v2.0 padrão: layer1(32,2,1), layer2(64,2,2), layer3(128,2,2)
        self.layer1 = self._make_layer(32, num_blocks=num_blocks, stride=1)
        self.layer2 = self._make_layer(64, num_blocks=num_blocks, stride=2)
        self.layer3 = self._make_layer(128, num_blocks=num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Cria um estágio com `num_blocks` blocos residuais.

        Args:
            out_channels: Canais de saída dos blocos.
            num_blocks: Número de blocos residuais no estágio.
            stride: Stride aplicado apenas no primeiro bloco (downsampling).

        Returns:
            Sequência de blocos residuais como nn.Sequential.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self._in_channels, out_channels, stride=s))
            self._in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagação forward."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
