"""Denoiser model zoo for plug-and-play deconvolution experiments.

This module intentionally ships *architectures and loading helpers only*.
It does not include third-party trained weights.  A model can be loaded from:

1. a raw PyTorch state_dict (.pt/.pth), selected by GUI model type; or
2. a checkpoint dictionary with key ``state_dict``; or
3. a sidecar JSON manifest next to the weights file, e.g. ``drunet_gray.json``.

Manifest example::

    {
      "architecture": "DRUNet",
      "channels": 1,
      "features": 64,
      "depth": 4,
      "residual": true,
      "clamp": true
    }

The implementations below are compact grayscale-friendly approximations of the
families used in plug-and-play reconstruction.  They are designed as stable
interfaces for loading compatible weights, not as a claim that bundled random
weights are useful denoisers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DenoiserConfig:
    architecture: str = "DnCNN"
    channels: int = 1
    features: int = 64
    depth: int = 17
    residual: bool = True
    clamp: bool = True
    noise_level: float = 0.05  # used by FFDNet-style models

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DenoiserConfig":
        data = asdict(cls())
        for k, v in d.items():
            if k in data:
                data[k] = v
        return cls(**data)


class ResidualWrapper(nn.Module):
    """Wrap a network that predicts either residual/noise or clean image."""
    def __init__(self, net: nn.Module, residual: bool = True, clamp: bool = True) -> None:
        super().__init__()
        self.net = net
        self.residual = residual
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        out = x - y if self.residual else y
        return torch.clamp(out, 0.0, 1.0) if self.clamp else out


class DnCNNCore(nn.Module):
    def __init__(self, channels: int = 1, features: int = 64, depth: int = 17) -> None:
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(features, features, 3, padding=1), nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FFDNetCore(nn.Module):
    """Small FFDNet-style model with an explicit noise-level map channel."""
    def __init__(self, channels: int = 1, features: int = 64, depth: int = 12, noise_level: float = 0.05) -> None:
        super().__init__()
        self.noise_level = float(noise_level)
        in_ch = channels + 1
        layers = [nn.Conv2d(in_ch, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.noise_level, dtype=x.dtype, device=x.device)
        return self.net(torch.cat([x, sigma], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DRUNetCore(nn.Module):
    """Compact DRUNet-like U-Net with residual blocks."""
    def __init__(self, channels: int = 1, features: int = 64, depth: int = 4) -> None:
        super().__init__()
        f = features
        self.in_conv = nn.Conv2d(channels, f, 3, padding=1)
        self.enc1 = nn.Sequential(ResidualBlock(f), ResidualBlock(f))
        self.down1 = nn.Conv2d(f, 2 * f, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(ResidualBlock(2 * f), ResidualBlock(2 * f))
        self.down2 = nn.Conv2d(2 * f, 4 * f, 3, stride=2, padding=1)
        self.mid = nn.Sequential(*[ResidualBlock(4 * f) for _ in range(max(1, depth))])
        self.up2 = nn.ConvTranspose2d(4 * f, 2 * f, 2, stride=2)
        self.dec2 = nn.Sequential(ResidualBlock(2 * f), ResidualBlock(2 * f))
        self.up1 = nn.ConvTranspose2d(2 * f, f, 2, stride=2)
        self.dec1 = nn.Sequential(ResidualBlock(f), ResidualBlock(f))
        self.out_conv = nn.Conv2d(f, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
        e1 = self.enc1(self.in_conv(xp))
        e2 = self.enc2(self.down1(e1))
        m = self.mid(self.down2(e2))
        d2 = self.up2(m)
        d2 = d2[..., :e2.shape[-2], :e2.shape[-1]] + e2
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = d1[..., :e1.shape[-2], :e1.shape[-1]] + e1
        y = self.out_conv(self.dec1(d1))
        return y[..., :h, :w]


class SCUNetCore(nn.Module):
    """Lightweight SCUNet-inspired convolution/attention denoiser.

    This is a practical placeholder interface for SCUNet-compatible experiments.
    It uses squeeze-excitation attention blocks; official SCUNet checkpoints may
    need a matching adapter/manifest if their layer names differ.
    """
    def __init__(self, channels: int = 1, features: int = 64, depth: int = 8) -> None:
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(1, depth)):
            layers.append(SEBlock(features))
        layers.append(nn.Conv2d(features, channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1))
        self.attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return x + y * self.attn(y)


def create_denoiser(name: str, **kwargs: Any) -> nn.Module:
    cfg = DenoiserConfig.from_dict({"architecture": name, **kwargs})
    arch = cfg.architecture.lower().replace("-", "").replace("_", "")
    if arch == "dncnn":
        core = DnCNNCore(cfg.channels, cfg.features, cfg.depth)
    elif arch == "ffdnet":
        core = FFDNetCore(cfg.channels, cfg.features, cfg.depth, cfg.noise_level)
    elif arch == "drunet":
        core = DRUNetCore(cfg.channels, cfg.features, cfg.depth)
    elif arch == "scunet":
        core = SCUNetCore(cfg.channels, cfg.features, cfg.depth)
    else:
        raise ValueError(f"Unknown denoiser architecture: {cfg.architecture}")
    return ResidualWrapper(core, residual=cfg.residual, clamp=cfg.clamp)


def _read_manifest(weights_path: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(weights_path)
    candidates = [path.with_suffix(".json"), path.parent / (path.stem + ".manifest.json")]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text(encoding="utf-8"))
    return None


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("The weights file must contain a state_dict or a checkpoint dict with key 'state_dict'.")
    return {str(k).replace("module.", "").replace("model.", ""): v for k, v in obj.items() if torch.is_tensor(v)}


def load_denoiser_from_file(weights_path: str | Path, architecture: Optional[str] = None, device: str | torch.device = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    manifest = _read_manifest(weights_path) or {}
    if architecture is not None:
        manifest["architecture"] = architecture
    if "architecture" not in manifest:
        manifest["architecture"] = "DnCNN"
    model = create_denoiser(**manifest).to(device)
    checkpoint = torch.load(str(weights_path), map_location=device)
    state = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    meta = dict(manifest)
    meta["missing_keys"] = list(missing)
    meta["unexpected_keys"] = list(unexpected)
    return model, meta


def save_manifest(path: str | Path, config: DenoiserConfig) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
