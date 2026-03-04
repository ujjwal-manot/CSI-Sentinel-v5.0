import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class CSIAugmentor(nn.Module):
    def __init__(
        self,
        time_mask_max: int = 30,
        freq_mask_max: int = 20,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        time_warp_max: int = 10,
        mixup_alpha: float = 0.2,
        cutout_ratio: float = 0.1,
        p: float = 0.5
    ):
        super().__init__()
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.time_warp_max = time_warp_max
        self.mixup_alpha = mixup_alpha
        self.cutout_ratio = cutout_ratio
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        if torch.rand(1).item() < self.p:
            x = self.time_mask(x)

        if torch.rand(1).item() < self.p:
            x = self.freq_mask(x)

        if torch.rand(1).item() < self.p:
            x = self.add_noise(x)

        if torch.rand(1).item() < self.p:
            x = self.random_scale(x)

        if torch.rand(1).item() < self.p * 0.5:
            x = self.cutout(x)

        return x

    def time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            _, h, w = x.shape
        else:
            _, _, h, w = x.shape

        mask_width = torch.randint(1, min(self.time_mask_max, w) + 1, (1,)).item()
        mask_start = torch.randint(0, w - mask_width + 1, (1,)).item()

        if x.dim() == 3:
            x = x.clone()
            x[:, :, mask_start:mask_start + mask_width] = 0
        else:
            x = x.clone()
            x[:, :, :, mask_start:mask_start + mask_width] = 0

        return x

    def freq_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            _, h, w = x.shape
        else:
            _, _, h, w = x.shape

        mask_height = torch.randint(1, min(self.freq_mask_max, h) + 1, (1,)).item()
        mask_start = torch.randint(0, h - mask_height + 1, (1,)).item()

        if x.dim() == 3:
            x = x.clone()
            x[:, mask_start:mask_start + mask_height, :] = 0
        else:
            x = x.clone()
            x[:, :, mask_start:mask_start + mask_height, :] = 0

        return x

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def random_scale(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        return x * scale

    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            _, h, w = x.shape
        else:
            _, _, h, w = x.shape

        cut_h = int(h * self.cutout_ratio)
        cut_w = int(w * self.cutout_ratio)

        cy = torch.randint(0, h, (1,)).item()
        cx = torch.randint(0, w, (1,)).item()

        y1 = max(0, cy - cut_h // 2)
        y2 = min(h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(w, cx + cut_w // 2)

        x = x.clone()
        if x.dim() == 3:
            x[:, y1:y2, x1:x2] = 0
        else:
            x[:, :, y1:y2, x1:x2] = 0

        return x

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            return x

        b, c, h, w = x.shape
        warp = torch.randint(-self.time_warp_max, self.time_warp_max + 1, (1,)).item()

        if warp == 0:
            return x

        if warp > 0:
            x = F.pad(x, (warp, 0), mode='replicate')[:, :, :, :w]
        else:
            x = F.pad(x, (0, -warp), mode='replicate')[:, :, :, -warp:]

        return x


class SpecAugment(nn.Module):
    def __init__(
        self,
        time_mask_param: int = 30,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        x = x.clone()

        if x.dim() == 3:
            c, h, w = x.shape
            batch = False
        else:
            b, c, h, w = x.shape
            batch = True

        for _ in range(self.num_time_masks):
            t = min(torch.randint(0, self.time_mask_param + 1, (1,)).item(), w)
            t0 = torch.randint(0, w - t + 1, (1,)).item()

            if batch:
                x[:, :, :, t0:t0 + t] = 0
            else:
                x[:, :, t0:t0 + t] = 0

        for _ in range(self.num_freq_masks):
            f = min(torch.randint(0, self.freq_mask_param + 1, (1,)).item(), h)
            f0 = torch.randint(0, h - f + 1, (1,)).item()

            if batch:
                x[:, :, f0:f0 + f, :] = 0
            else:
                x[:, f0:f0 + f, :] = 0

        return x


class MixUp:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]

        return mixed_x, y, y[index], lam


class CutMix:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        _, _, h, w = x.shape

        cut_rat = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)

        cy = np.random.randint(h)
        cx = np.random.randint(w)

        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)

        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))

        return mixed_x, y, y[index], lam


def create_transform_pipeline(config: dict) -> CSIAugmentor:
    return CSIAugmentor(
        time_mask_max=config.get("time_mask_max", 30),
        freq_mask_max=config.get("freq_mask_max", 20),
        noise_std=config.get("noise_std", 0.02),
        scale_range=tuple(config.get("scale_range", [0.8, 1.2])),
        p=config.get("augmentation_probability", 0.5)
    )
