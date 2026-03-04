import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class RFEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = True
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TemporalAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)

        qkv = self.qkv(x_norm).reshape(B, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)

        out = out + x_flat
        out = out.transpose(1, 2).reshape(B, C, H, W)

        return out


class RFEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
        embedding_dim: int = 512,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()

        self.input_channels = input_channels
        self.embedding_dim = embedding_dim

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.ModuleList()
        in_channels = base_channels

        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            stride = 1 if i == 0 else 2

            downsample = None
            if stride != 1 or in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

            self.layers.append(RFEncoderBlock(in_channels, out_channels, stride, downsample))
            self.layers.append(RFEncoderBlock(out_channels, out_channels))

            in_channels = out_channels

        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(in_channels, num_heads=8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)

        if self.use_attention:
            x = self.attention(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)

        x = F.normalize(x, p=2, dim=-1)

        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        features = []

        x = self.stem(x)
        features.append(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        if self.use_attention:
            x = self.attention(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        embedding = self.fc(x)
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding, features


class RFEncoderSmall(RFEncoder):
    def __init__(self, input_channels: int = 3, embedding_dim: int = 256, dropout: float = 0.1):
        super().__init__(
            input_channels=input_channels,
            base_channels=32,
            num_blocks=3,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_attention=False
        )


class RFEncoderLarge(RFEncoder):
    def __init__(self, input_channels: int = 3, embedding_dim: int = 768, dropout: float = 0.1):
        super().__init__(
            input_channels=input_channels,
            base_channels=64,
            num_blocks=5,
            embedding_dim=embedding_dim,
            dropout=dropout,
            use_attention=True
        )
