"""Numerical data models and shared deconvolution utilities.

This module contains reusable numerical infrastructure only.  It has no Qt GUI
imports and does not define concrete deconvolution algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import rotate, gaussian_filter, shift as ndimage_shift, uniform_filter
from scipy.io import loadmat
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_nl_means, denoise_wavelet, estimate_sigma

from .operators import (
    LINEAR_SAME, CIRCULAR_FFT, NumpyLinearSameOperator, TorchLinearSameOperator,
    psf_at_fft_origin_numpy, psf_to_otf_numpy, psf_to_otf_torch,
    circular_convolve_numpy, circular_convolve_torch,
    linear_convolve_same_numpy, linear_correlate_same_numpy, linear_convolve_same_torch,
)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    F = None
    TORCH_AVAILABLE = False



_IMAGE_DATA_STATE_KEYS = {
    "image",
    "degraded",
    "psf",
    "calculation_psf",
    "degradation_psf",
    "result",
    "estimated_psf",
    "last_run_psf",
    "last_run_calculation_shape",
    "last_shape_reconciliation",
    "measured_pair_loaded",
    "reference_available",
    "reference_source",
    "psf_automatic_selection",
    "psf_floor_wiener_optimization",
    "optimized_wiener_k",
    "psf_calculation_center_mode",
    "psf_calculation_center_x",
    "psf_calculation_center_y",
    "psf_selection_generation",
    "psf_support_extent",
    "psf_support_hard_cap",
    "psf_support_height",
    "psf_support_width",
    "limit_psf_support",
    "_tab2_threshold_base_degraded",
    "_tab2_threshold_base_psf",
    "_tab2_threshold_base_degradation_psf",
    "_tab2_threshold_base_psf_selection",
}


def clear_image_data_state(state: Dict[str, Any]) -> None:
    """Remove loaded/generated images, PSFs and derived numerical products.

    GUI and algorithm settings are intentionally preserved. In particular, the
    selected calculation resolution, zero-padding mode and Wiener profile remain
    unchanged, so another data set can be loaded without rebuilding the
    numerical configuration.
    """
    for key in _IMAGE_DATA_STATE_KEYS:
        state.pop(key, None)
    state["reference_available"] = False
    state["reference_source"] = None
    state["measured_pair_loaded"] = False


def mat_array_candidates(path: str) -> Dict[str, np.ndarray]:
    """Return numeric two-dimensional arrays available in a MATLAB MAT file."""
    raw = loadmat(path)
    candidates: Dict[str, np.ndarray] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        arr = np.squeeze(arr)
        if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number) and arr.size > 0:
            candidates[key] = arr
    return candidates


def _array_to_unit_float(arr: np.ndarray) -> np.ndarray:
    """Convert a monochrome numeric array to finite float64 values in [0, 1].

    Integer images retain their encoded dynamic range (8-bit, 16-bit, etc.).
    Floating-point arrays already in [0, 1] are preserved; other floating-point
    ranges are linearly mapped to [0, 1].
    """
    source = np.asarray(arr)
    source = np.squeeze(source)
    if source.ndim == 3:
        if source.shape[-1] in (3, 4):
            rgb = source[..., :3].astype(np.float64)
            source = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        elif 1 in source.shape:
            source = np.squeeze(source)
    if source.ndim != 2:
        raise ValueError("The selected data must be a two-dimensional monochrome image.")

    if np.issubdtype(source.dtype, np.bool_):
        out = source.astype(np.float64)
    elif np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        out = source.astype(np.float64)
        if info.min < 0:
            out = out - float(info.min)
            scale = float(info.max) - float(info.min)
        else:
            scale = float(info.max)
        if scale > 0.0:
            out /= scale
    else:
        out = source.astype(np.float64)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        mn, mx = float(out.min()), float(out.max())
        if mn < 0.0 or mx > 1.0:
            if mx > mn:
                out = (out - mn) / (mx - mn)
            elif mx != 0.0:
                out = out / mx
    return np.clip(np.nan_to_num(out), 0.0, 1.0)


def load_monochrome_array(
    path: str,
    mat_key: Optional[str] = None,
    preferred_mat_keys: Tuple[str, ...] = (),
) -> np.ndarray:
    """Load an 8/16-bit monochrome image or a 2D array from a MAT file."""
    suffix = Path(path).suffix.lower()
    if suffix == ".mat":
        candidates = mat_array_candidates(path)
        if not candidates:
            raise ValueError("The MAT file contains no numeric two-dimensional arrays.")
        key = mat_key
        if key is None:
            for preferred in preferred_mat_keys:
                if preferred in candidates:
                    key = preferred
                    break
        if key is None:
            key = next(iter(candidates))
        if key not in candidates:
            raise KeyError(f"MAT variable '{key}' is not a numeric two-dimensional array.")
        return _array_to_unit_float(candidates[key])

    with Image.open(path) as img:
        # np.asarray preserves 16-bit PNG/TIFF data. Image.convert('L') would
        # quantize it to eight bits and is therefore deliberately avoided.
        arr = np.asarray(img)
    return _array_to_unit_float(arr)

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default

@dataclass
class GrayImage:
    """Normalized 2D grayscale image stored as a float64 array in the [0, 1] range."""

    data: np.ndarray
    name: str = "image"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if bool(self.metadata.get("_preserve_intensity", False)):
            arr = np.asarray(self.data, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError("Image must be a 2D grayscale array.")
            self.data = np.clip(np.nan_to_num(arr), 0.0, 1.0)
        else:
            self.data = self._normalize_array(self.data)

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Image must be a 2D grayscale array.")
        arr = np.nan_to_num(arr)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def resize_array(arr: np.ndarray, width: int = 256, height: int = 256) -> np.ndarray:
        """Resize an array with PIL bilinear interpolation and keep values in [0, 1]."""
        arr = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img = img.resize((int(width), int(height)), Image.BILINEAR)
        return np.asarray(img, dtype=np.float64) / 255.0

    @staticmethod
    def _normalize_content(arr: np.ndarray) -> np.ndarray:
        """Normalize source content before it is placed inside a zero frame."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("Image must be a 2D grayscale array.")
        arr = np.nan_to_num(arr)
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        return np.clip(arr, 0.0, 1.0)

    @classmethod
    def from_array_with_zero_frame(
        cls,
        source: np.ndarray,
        width: int = 256,
        height: int = 256,
        padding: int = 0,
        name: str = "image",
    ) -> "GrayImage":
        """Resize source into a centered zero frame of the requested calculation size.

        For an odd PSF support K, padding K//2 leaves an inner image of
        N-K+1 pixels. Then the full linear convolution of the non-zero content
        with the PSF fits inside the N x N calculation image, while FFT/same
        convolution does not wrap or truncate the physically relevant result.
        """
        width = int(width)
        height = int(height)
        padding = int(max(0, padding))
        padding = min(padding, max(0, (width - 1) // 2), max(0, (height - 1) // 2))
        inner_w = max(1, width - 2 * padding)
        inner_h = max(1, height - 2 * padding)
        src = cls._normalize_content(source)
        content = cls.resize_array(src, inner_w, inner_h)
        framed = np.zeros((height, width), dtype=np.float64)
        y0 = (height - inner_h) // 2
        x0 = (width - inner_w) // 2
        framed[y0:y0 + inner_h, x0:x0 + inner_w] = content
        return cls(
            framed,
            name=name,
            metadata={
                "calculation_size": (height, width),
                "zero_padding": padding,
                "inner_size": (inner_h, inner_w),
                "content_roi": (y0, y0 + inner_h, x0, x0 + inner_w),
                "source_array": src,
            },
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        width: int = 256,
        height: int = 256,
        padding: int = 0,
        mat_key: Optional[str] = None,
    ) -> "GrayImage":
        arr = load_monochrome_array(
            path,
            mat_key=mat_key,
            preferred_mat_keys=("degraded", "measured", "image", "result", "reference"),
        )
        return cls.from_array_with_zero_frame(arr, width=width, height=height, padding=padding, name=path)

    @staticmethod
    def _hidden_phrase() -> str:
        word1 = [21, 14, 9, 22, 5, 18, 19, 9, 20, 25]
        word2 = [23, 1, 18, 19, 1, 23]
        part1 = ''.join(chr(64 + x) for x in word1).title()
        part2 = ''.join(chr(64 + x) for x in word2).title()
        return part1 + ' ' + part2

    @classmethod
    def _embed_hidden_phrase(cls, arr: np.ndarray) -> np.ndarray:
        """Embed a clearly visible phrase rendered letter-by-letter on a warped path.

        The phrase is reconstructed from numeric codes rather than stored as a
        plain text literal, so it is still difficult to find and modify by a
        simple text search in the source code.
        """
        h, w = arr.shape
        base = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), mode='L').convert('RGBA')
        phrase = cls._hidden_phrase()
        font_size = max(12, int(min(h, w) * 0.065))
        try:
            font = ImageFont.truetype('DejaVuSans-Bold.ttf', font_size)
        except Exception:
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', font_size)
            except Exception:
                font = ImageFont.load_default()

        chars = [ch for ch in phrase]
        t = np.linspace(-1.08, 1.08, len(chars))
        cx, cy = 0.50 * w, 0.56 * h
        rx, ry = 0.245 * w, 0.155 * h

        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        for idx, (u, ch) in enumerate(zip(t, chars)):
            if ch == ' ':
                continue
            x = cx + rx * u + 0.025 * w * np.sin(2.5 * np.pi * u)
            y = cy + ry * np.sin(0.95 * np.pi * u) + 0.025 * h * np.cos(1.7 * np.pi * u)
            angle = -36.0 + 72.0 * ((idx / max(1, len(chars) - 1)) - 0.5) + 10.0 * np.sin(2.8 * u)
            char_img = Image.new('RGBA', (font_size * 4, font_size * 4), (0, 0, 0, 0))
            d = ImageDraw.Draw(char_img)
            # Two-pass render: soft dark halo + bright core for readability on any background.
            d.text((2 * font_size + 1, 2 * font_size + 1), ch, font=font, fill=(10, 10, 10, 185), anchor='mm')
            d.text((2 * font_size, 2 * font_size), ch, font=font, fill=(245, 245, 245, 235), anchor='mm')
            char_img = char_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            px = int(round(x - char_img.width / 2))
            py = int(round(y - char_img.height / 2))
            overlay.alpha_composite(char_img, dest=(px, py))

        merged = Image.alpha_composite(base, overlay)
        out = np.asarray(merged.convert('L'), dtype=np.float64) / 255.0
        # Stronger blend than before so the phrase is clearly visible in the test image.
        return np.clip(0.68 * arr + 0.32 * out, 0.0, 1.0)

    @classmethod
    def synthetic(
        cls,
        width: int = 256,
        height: Optional[int] = 256,
        padding: int = 0,
        internal_margin_fraction: float = 0.15,
    ) -> "GrayImage":
        """Generate a rectangular test image with broad internal margins.

        ``width`` and ``height`` describe the calculation canvas.  The synthetic
        features occupy only the central part of the source image, leaving about
        15 percent of each dimension as an intrinsic dark margin.  Consequently,
        the default test image normally does not require an additional visible
        zero frame for moderate PSF supports.

        ``height=None`` preserves compatibility with older square-image calls.
        """
        width = int(width)
        height = int(width if height is None else height)
        padding = int(max(0, padding))
        inner_w = max(8, width - 2 * padding)
        inner_h = max(8, height - 2 * padding)
        arr = np.zeros((inner_h, inner_w), dtype=np.float64)

        margin_fraction = float(np.clip(internal_margin_fraction, 0.05, 0.30))
        mx = max(8, int(round(inner_w * margin_fraction)))
        my = max(8, int(round(inner_h * margin_fraction)))
        x0, x1 = mx, max(mx + 1, inner_w - mx)
        y0, y1 = my, max(my + 1, inner_h - my)
        content_w = max(1, x1 - x0)
        content_h = max(1, y1 - y0)

        # Central rectangular feature.
        ry0 = y0 + int(0.22 * content_h)
        ry1 = y0 + int(0.80 * content_h)
        rx0 = x0 + int(0.31 * content_w)
        rx1 = x0 + int(0.68 * content_w)
        arr[ry0:ry1, rx0:rx1] = 0.70

        # Circular feature, scaled by the shorter content dimension.
        rr, cc = np.ogrid[:inner_h, :inner_w]
        cy = y0 + content_h // 2
        cx = x0 + content_w // 2
        radius = max(3, int(0.15 * min(content_h, content_w)))
        circle = (rr - cy) ** 2 + (cc - cx) ** 2 < radius ** 2
        arr[circle] = 1.0

        # Long horizontal feature, still well inside the intrinsic margins.
        by0 = y0 + int(0.06 * content_h)
        by1 = y0 + max(int(0.15 * content_h), int(0.06 * content_h) + 1)
        bx0 = x0 + int(0.06 * content_w)
        bx1 = x0 + int(0.94 * content_w)
        arr[by0:by1, bx0:bx1] = 0.35

        # The phrase remains within the central region and is rendered one
        # character at a time using the existing indirect text representation.
        arr = cls._embed_hidden_phrase(arr)
        result = cls.from_array_with_zero_frame(
            arr,
            width=width,
            height=height,
            padding=padding,
            name="synthetic",
        )
        result.metadata.update({
            "synthetic_internal_margin_fraction": margin_fraction,
            "synthetic_internal_margin_xy": (mx, my),
            "synthetic_feature_roi": (y0, y1, x0, x1),
        })
        return result

    def save(self, path: str) -> None:
        Image.fromarray((np.clip(self.data, 0, 1) * 255).astype(np.uint8)).save(path)

@dataclass
class PSF:
    """Point spread function with raw and calculation kernels.

    raw_kernel preserves the loaded/generated intensity map for inspection.
    kernel is the working non-negative, sum-normalized convolution kernel used
    by all algorithms.
    """

    kernel: np.ndarray
    name: str = "psf"
    raw_kernel: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = np.asarray(self.kernel if self.raw_kernel is None else self.raw_kernel, dtype=np.float64)
        raw = np.nan_to_num(raw)
        raw = np.maximum(raw, 0.0)
        self.raw_kernel = raw.copy()

        arr = np.asarray(self.kernel, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("PSF must be a 2D array.")
        self.kernel = self.normalize_kernel(arr)

    @staticmethod
    def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
        """Return a non-negative PSF with unit sum."""
        arr = np.asarray(kernel, dtype=np.float64)
        arr = np.nan_to_num(arr)
        arr = np.maximum(arr, 0.0)
        s = float(arr.sum())
        if s <= 0:
            raise ValueError("PSF sum must be positive.")
        return arr / s

    @classmethod
    def from_file(
        cls,
        path: str,
        target_size: Optional[int] = None,
        mat_key: Optional[str] = None,
    ) -> "PSF":
        """Load an 8/16-bit PSF image or a two-dimensional MAT variable.

        The loaded data are not support-cropped here. The calculation-safe PSF
        is produced by fitted_to_shape() just before convolution/deconvolution.
        """
        arr = load_monochrome_array(
            path,
            mat_key=mat_key,
            preferred_mat_keys=("psf", "psf_kernel", "current_psf", "degradation_psf", "estimated_psf"),
        )
        if target_size is not None and target_size > 0:
            target_size = int(target_size)
            arr = GrayImage.resize_array(arr, target_size, target_size)
        arr = np.maximum(arr, 0.0)
        return cls(arr, name=path, raw_kernel=arr)

    @staticmethod
    def center_of_mass(kernel: np.ndarray, threshold_fraction: float = 1e-3) -> Tuple[float, float]:
        """Estimate PSF center using a thresholded center of mass.

        The threshold suppresses a large dark background in PSF images loaded
        from disk. If the center of mass is ill-defined, the maximum is used.
        """
        arr = np.asarray(kernel, dtype=np.float64)
        arr = np.nan_to_num(arr)
        arr = np.maximum(arr, 0.0)
        if arr.size == 0 or float(arr.max()) <= 0.0:
            return (arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0
        peak = np.unravel_index(int(np.argmax(arr)), arr.shape)
        weights = arr * (arr >= float(arr.max()) * float(threshold_fraction))
        total = float(weights.sum())
        if total <= 0.0:
            return float(peak[0]), float(peak[1])
        yy, xx = np.indices(arr.shape)
        cy = float((yy * weights).sum() / total)
        cx = float((xx * weights).sum() / total)
        cy = float(np.clip(cy, 0, arr.shape[0] - 1))
        cx = float(np.clip(cx, 0, arr.shape[1] - 1))
        return cy, cx

    @staticmethod
    def support_center(kernel: np.ndarray, threshold_fraction: float = 1e-3) -> Tuple[int, int]:
        cy, cx = PSF.center_of_mass(kernel, threshold_fraction=threshold_fraction)
        return int(round(cy)), int(round(cx))

    @staticmethod
    def automatic_support_selection(
        kernel: np.ndarray,
        peak_fraction: float = 1e-2,
        border_sigma: float = 3.0,
        margin: int = 2,
    ) -> Dict[str, Any]:
        """Estimate a useful rectangular PSF calculation window.

        A robust background and noise level are inferred from perimeter pixels.
        The active support contains pixels above both the border-noise threshold
        and ``peak_fraction`` of the *raw PSF maximum*.  The default fraction is
        1e-2, deliberately excluding very weak tails from the initial proposal.
        Width and height are estimated independently, so elongated motion or
        astigmatic PSFs are not forced into a square crop.  The result remains a
        starting point that the user can edit before applying it.
        """
        arr = np.asarray(kernel, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return {
                "center": (0, 0), "height": 1, "width": 1,
                "background": 0.0, "threshold": 0.0,
                "floor_fraction": 0.0, "active_pixels": 0,
            }
        arr = np.maximum(np.nan_to_num(arr), 0.0)
        h, w = arr.shape
        if h == 1 or w == 1:
            border = arr.ravel()
        else:
            border = np.concatenate((arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]))
        border = border[np.isfinite(border)]
        background = float(np.median(border)) if border.size else 0.0
        mad = float(np.median(np.abs(border - background))) if border.size else 0.0
        noise_sigma = 1.4826 * mad
        raw_peak = float(np.max(arr)) if arr.size else 0.0
        residual = np.maximum(arr - background, 0.0)
        if raw_peak <= 1e-18 or float(np.max(residual)) <= 1e-18:
            py, px = np.unravel_index(int(np.argmax(arr)), arr.shape)
            return {
                "center": (int(py), int(px)), "height": 1, "width": 1,
                "background": background, "threshold": background,
                "floor_fraction": background / raw_peak if raw_peak > 1e-18 else 0.0,
                "active_pixels": 1,
            }

        absolute_threshold = max(
            background + float(border_sigma) * noise_sigma,
            float(peak_fraction) * raw_peak,
        )
        active = arr > absolute_threshold
        if not np.any(active):
            active[np.unravel_index(int(np.argmax(arr)), arr.shape)] = True
        weights = np.where(active, residual, 0.0)
        total = float(weights.sum())
        if total > 1e-18:
            yy, xx = np.indices(arr.shape)
            cy_f = float((yy * weights).sum() / total)
            cx_f = float((xx * weights).sum() / total)
        else:
            cy_f, cx_f = map(float, np.unravel_index(int(np.argmax(arr)), arr.shape))
        cy = int(np.clip(round(cy_f), 0, h - 1))
        cx = int(np.clip(round(cx_f), 0, w - 1))
        ys, xs = np.nonzero(active)
        radius_y = int(np.ceil(float(np.max(np.abs(ys - cy))) if ys.size else 0.0)) + max(0, int(margin))
        radius_x = int(np.ceil(float(np.max(np.abs(xs - cx))) if xs.size else 0.0)) + max(0, int(margin))
        height = min(max(1, 2 * radius_y + 1), h)
        width = min(max(1, 2 * radius_x + 1), w)
        if height > 1 and height % 2 == 0:
            height -= 1
        if width > 1 and width % 2 == 0:
            width -= 1
        return {
            "center": (cy, cx),
            "height": int(max(1, height)),
            "width": int(max(1, width)),
            "background": float(background),
            "border_sigma": float(noise_sigma),
            "threshold": float(absolute_threshold),
            "floor_fraction": float(np.clip(absolute_threshold / raw_peak, 0.0, 1.0)),
            "active_pixels": int(np.count_nonzero(active)),
            "peak_fraction": float(peak_fraction),
        }

    @staticmethod
    def centered_window(kernel: np.ndarray, center: Tuple[int, int], height: int, width: int) -> np.ndarray:
        """Extract a window centered on center, padding with zeros if needed."""
        arr = np.asarray(kernel, dtype=np.float64)
        height = max(1, int(height))
        width = max(1, int(width))
        out = np.zeros((height, width), dtype=np.float64)
        cy, cx = int(center[0]), int(center[1])
        y0 = cy - height // 2
        x0 = cx - width // 2
        y1 = y0 + height
        x1 = x0 + width
        sy0 = max(0, y0)
        sx0 = max(0, x0)
        sy1 = min(arr.shape[0], y1)
        sx1 = min(arr.shape[1], x1)
        dy0 = sy0 - y0
        dx0 = sx0 - x0
        if sy1 > sy0 and sx1 > sx0:
            out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
        return out

    def fitted_to_shape(self, shape: Tuple[int, int], max_width: Optional[int] = None) -> "PSF":
        """Return a calculation-safe PSF for images of shape.

        The support is extracted around the centre selected in PSF metadata:
        center of mass, geometric centre, or an explicit manual pixel. This
        matters for imported PSF images whose useful support is shifted inside
        a larger frame. The returned PSF is normalized.
        """
        h, w = int(shape[0]), int(shape[1])
        arr = np.asarray(self.kernel, dtype=np.float64)

        source_metadata = dict(self.metadata or {})
        requested_h = source_metadata.get("calculation_support_height")
        requested_w = source_metadata.get("calculation_support_width")
        has_rectangular_selection = requested_h is not None or requested_w is not None
        if has_rectangular_selection:
            requested_h = int(requested_h if requested_h is not None else requested_w)
            requested_w = int(requested_w if requested_w is not None else requested_h)
            target_h = min(max(1, requested_h), arr.shape[0], h)
            target_w = min(max(1, requested_w), arr.shape[1], w)
            if max_width is not None and int(max_width) > 0:
                global_cap = max(1, int(max_width))
                target_h = min(target_h, global_cap)
                target_w = min(target_w, global_cap)
        elif max_width is None or int(max_width) <= 0:
            target_h = min(arr.shape[0], h)
            target_w = min(arr.shape[1], w)
        else:
            mw = int(max_width)
            if mw % 2 == 0:
                mw -= 1
            mw = max(3, mw)
            target_h = min(mw, h)
            target_w = min(mw, w)

        if not has_rectangular_selection:
            if target_h > 1 and target_h % 2 == 0:
                target_h -= 1
            if target_w > 1 and target_w % 2 == 0:
                target_w -= 1
        target_h = max(1, target_h)
        target_w = max(1, target_w)

        center_mode = str(source_metadata.get("calculation_center_mode", "center_of_mass"))
        if center_mode == "geometric":
            center = (arr.shape[0] // 2, arr.shape[1] // 2)
        elif center_mode == "manual":
            requested = source_metadata.get("calculation_center")
            if isinstance(requested, (tuple, list)) and len(requested) == 2:
                center = (
                    int(np.clip(int(round(float(requested[0]))), 0, max(0, arr.shape[0] - 1))),
                    int(np.clip(int(round(float(requested[1]))), 0, max(0, arr.shape[1] - 1))),
                )
            else:
                center = self.support_center(arr)
                center_mode = "center_of_mass"
        else:
            requested = source_metadata.get("calculation_center")
            if isinstance(requested, (tuple, list)) and len(requested) == 2:
                center = (
                    int(np.clip(int(round(float(requested[0]))), 0, max(0, arr.shape[0] - 1))),
                    int(np.clip(int(round(float(requested[1]))), 0, max(0, arr.shape[1] - 1))),
                )
            else:
                center = self.support_center(arr)
            center_mode = "center_of_mass"
        cropped = self.centered_window(arr, center, target_h, target_w)
        selection_empty_fallback = False
        if float(np.sum(np.maximum(np.nan_to_num(cropped), 0.0))) <= 1e-18:
            # A user-selected geometric window may miss a strongly displaced
            # measured PSF completely. Keep the convolution operator valid and
            # make the fallback explicit in metadata instead of failing later.
            cropped = np.zeros((target_h, target_w), dtype=np.float64)
            cropped[target_h // 2, target_w // 2] = 1.0
            selection_empty_fallback = True
        metadata = dict(self.metadata or {})
        metadata.update({
            "source_psf_name": self.name,
            "source_psf_shape": tuple(int(v) for v in arr.shape),
            "source_psf_center": tuple(int(v) for v in center),
            "source_calculation_center_mode": center_mode,
            # The selected source centre is mapped to the geometric centre of
            # the cropped kernel.  Keeping this local coordinate makes a later
            # emergency refit of an already fitted PSF safe.
            "calculation_center_mode": "manual" if center_mode == "manual" else center_mode,
            "calculation_center": (int(target_h // 2), int(target_w // 2)),
            "fitted_support": (int(target_h), int(target_w)),
            "calculation_support_height": int(target_h),
            "calculation_support_width": int(target_w),
            "empty_selection_replaced_by_impulse": bool(selection_empty_fallback),
        })
        return PSF(
            cropped,
            name=self.name + f"_support_{target_h}x{target_w}",
            raw_kernel=self.raw_kernel,
            metadata=metadata,
        )

    @classmethod
    def gaussian(cls, size: int = 21, sigma: float = 3.0) -> "PSF":
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return cls(kernel, name=f"gaussian_{size}_{sigma}")

    @classmethod
    def motion(cls, size: int = 21, angle_deg: float = 0.0) -> "PSF":
        """Motion blur PSF. A non-zero angle creates an oblique motion PSF."""
        kernel = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        kernel[center, :] = 1.0
        if abs(angle_deg) > 1e-9:
            kernel = rotate(kernel, angle=angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
            kernel = np.maximum(kernel, 0.0)
        return cls(kernel, name=f"motion_{size}_{angle_deg}")

    @classmethod
    def high_frequency(cls, size: int = 21, frequency: float = 4.0, sigma: float = 4.0) -> "PSF":
        """Oscillatory high-frequency PSF useful for stress-testing algorithms."""
        ax = np.arange(-(size // 2), size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        envelope = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        carrier = 1.0 + 0.75 * np.cos(2.0 * np.pi * frequency * xx / max(size, 1)) * np.cos(2.0 * np.pi * frequency * yy / max(size, 1))
        kernel = np.maximum(envelope * carrier, 0.0)
        return cls(kernel, name=f"high_frequency_{size}_{frequency}_{sigma}")

    @staticmethod
    def rotational_average(kernel: np.ndarray, radial_bins: Optional[int] = None, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Make a PSF rotationally invariant by radial averaging.

        When center is not provided, the radial center is estimated as the PSF
        center of mass instead of assuming the geometric center of the image.
        """
        arr = np.asarray(kernel, dtype=np.float64)
        arr = np.nan_to_num(arr)
        arr = np.maximum(arr, 0.0)
        if arr.ndim != 2:
            raise ValueError("PSF must be a 2D array.")
        if center is None:
            cy, cx = PSF.center_of_mass(arr)
        else:
            cy, cx = float(center[0]), float(center[1])
        yy, xx = np.indices(arr.shape)
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        if radial_bins is None:
            rb = np.rint(r).astype(int)
        else:
            rb = np.minimum((r / max(float(r.max()), 1e-12) * (radial_bins - 1)).astype(int), radial_bins - 1)
        sums = np.bincount(rb.ravel(), weights=arr.ravel())
        counts = np.bincount(rb.ravel())
        radial = sums / np.maximum(counts, 1)
        averaged = radial[rb]
        averaged = np.maximum(averaged, 0.0)
        return PSF.normalize_kernel(averaged)


    @staticmethod
    def recenter_to_geometric_center(kernel: np.ndarray) -> np.ndarray:
        """Shift a PSF so that its center of mass is at the geometric center.

        Blind deconvolution has an inherent shift ambiguity: the estimated image
        and PSF can compensate for each other by opposite shifts.  When a
        rotationally symmetric PSF is requested, the symmetry should be around
        the PSF array center used by the convolution operator, not around a
        drifting off-center center of mass.
        """
        arr = np.asarray(kernel, dtype=np.float64)
        arr = np.nan_to_num(arr)
        arr = np.maximum(arr, 0.0)
        if arr.ndim != 2:
            raise ValueError("PSF must be a 2D array.")
        cy, cx = PSF.center_of_mass(arr)
        target_y = (arr.shape[0] - 1) / 2.0
        target_x = (arr.shape[1] - 1) / 2.0
        shifted = ndimage_shift(
            arr,
            shift=(target_y - cy, target_x - cx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        return PSF.normalize_kernel(np.maximum(shifted, 0.0))

    @staticmethod
    def rotational_project_centered(kernel: np.ndarray) -> np.ndarray:
        """Project PSF to centered, nonnegative, sum-one radial symmetry."""
        arr = PSF.recenter_to_geometric_center(kernel)
        center = ((arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0)
        arr = PSF.rotational_average(arr, center=center)
        arr = PSF.recenter_to_geometric_center(arr)
        center = ((arr.shape[0] - 1) / 2.0, (arr.shape[1] - 1) / 2.0)
        arr = PSF.rotational_average(arr, center=center)
        return PSF.normalize_kernel(np.maximum(arr, 0.0))

    @staticmethod
    def central_crop(kernel: np.ndarray, crop_size: int, center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Return a crop used only for display."""
        arr = np.asarray(kernel)
        crop_size = max(1, min(int(crop_size), arr.shape[0], arr.shape[1]))
        if crop_size % 2 == 0:
            crop_size -= 1
        if center is None:
            cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        else:
            cy, cx = int(center[0]), int(center[1])
        half = crop_size // 2
        return PSF.centered_window(arr, (cy, cx), crop_size, crop_size)


    @classmethod
    def lens_incoherent(
        cls,
        size: int = 65,
        focal_length: float = 0.05,
        distance_before: float = 0.10,
        distance_after: float = 0.10,
        wavelength: float = 550e-9,
        aperture_diameter: float = 0.005,
        diffraction_grid_size: Optional[int] = None,
    ) -> "PSF":
        """Approximate incoherent PSF for a thin lens using an oversized diffraction grid.

        Parameters are in SI units. The coherent amplitude PSF is approximated by
        the Fourier transform of a circular pupil with quadratic defocus phase;
        the incoherent PSF is the squared magnitude of that complex field. The
        diffraction grid is at least twice as large as the requested PSF crop.
        """
        if size % 2 == 0:
            size += 1
        n = int(diffraction_grid_size or max(2 * size, 129))
        if n < 2 * size:
            n = 2 * size
        if n % 2 == 0:
            n += 1
        x = np.linspace(-aperture_diameter / 2.0, aperture_diameter / 2.0, n)
        xx, yy = np.meshgrid(x, x)
        r2 = xx ** 2 + yy ** 2
        aperture = (r2 <= (aperture_diameter / 2.0) ** 2).astype(np.float64)
        k = 2.0 * np.pi / max(wavelength, 1e-12)
        defocus = (1.0 / max(distance_before, 1e-12)) + (1.0 / max(distance_after, 1e-12)) - (1.0 / max(focal_length, 1e-12))
        pupil = aperture * np.exp(1j * 0.5 * k * defocus * r2)
        field = fftshift(fft2(pupil))
        intensity = np.abs(field) ** 2
        intensity = cls.central_crop(intensity, size)
        return cls(intensity, name=f"lens_incoherent_grid_{n}")

def _match_array_shape(arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Return arr resized to shape using bilinear interpolation when needed."""
    data = np.asarray(arr, dtype=np.float64)
    if data.shape == tuple(shape):
        return data
    data = np.nan_to_num(data)
    if data.size == 0:
        return np.zeros(shape, dtype=np.float64)
    mn, mx = float(data.min()), float(data.max())
    if mx > mn:
        norm = (data - mn) / (mx - mn)
    else:
        norm = np.zeros_like(data, dtype=np.float64)
    img = Image.fromarray((np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8))
    img = img.resize((int(shape[1]), int(shape[0])), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0

def _odd_at_most(value: int, minimum: int = 3) -> int:
    """Return an odd integer not larger than value."""
    value = int(value)
    if value % 2 == 0:
        value -= 1
    return max(int(minimum), value)

def resolution_linked_psf_support(
    image_shape: Tuple[int, int],
    fraction: float = 0.45,
    minimum: int = 3,
) -> int:
    """Return an odd square PSF support linked to image resolution.

    The support is computed from ``fraction`` of the smaller image dimension
    and is additionally limited by :func:`max_psf_support_for_image`.  Using
    the smaller dimension keeps a square PSF inside both image directions.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    frac = float(np.clip(fraction, 0.001, 0.5))
    proposed = max(int(minimum), int(round(frac * min(h, w))))
    return _odd_at_most(min(proposed, max_psf_support_for_image((h, w))), minimum=minimum)

def max_psf_support_for_image(image_shape: Tuple[int, int]) -> int:
    """Maximum selected PSF extent that fits on the calculation image.

    Tab 2 is authoritative for known-PSF support.  A user may intentionally
    select the complete full-resolution PSF array, including even dimensions,
    so the safety cap is the smaller image dimension rather than half of it.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    return max(1, min(h, w))

def degradation_kernel_for_image(psf: PSF, image_shape: Tuple[int, int], max_width: Optional[int] = None) -> PSF:
    """Create the single authoritative PSF used to generate degraded data.

    This helper is the only place where the source/display PSF is cropped,
    centered, support-limited and normalized for degradation.  The returned
    object must be stored in state["degradation_psf"] and reused unchanged by
    non-blind reconstruction algorithms.

    Safety rule: the calculation PSF support may not exceed half of the image
    width or height. This keeps enough zero-framed object area for linear
    convolution and prevents extremely wide PSFs from collapsing the usable
    image content.
    """
    if not isinstance(psf, PSF):
        psf = PSF(psf)
    hard_cap = max_psf_support_for_image(image_shape)
    metadata = dict(getattr(psf, "metadata", {}) or {})
    requested_h = metadata.get("calculation_support_height")
    requested_w = metadata.get("calculation_support_width")
    if requested_h is not None or requested_w is not None:
        # The Tab-2 rectangle is authoritative and may span the complete,
        # non-square image. Per-axis clipping is already performed by
        # ``fitted_to_shape``; a legacy scalar cap must not shorten its longer
        # side.
        requested_h = int(requested_h if requested_h is not None else requested_w)
        requested_w = int(requested_w if requested_w is not None else requested_h)
        safe_width = None
    elif max_width is None or int(max_width) <= 0:
        safe_width = hard_cap
    else:
        safe_width = min(max(1, int(max_width)), hard_cap)
    fitted = psf.fitted_to_shape(image_shape, max_width=safe_width)
    metadata = dict(fitted.metadata or {})
    metadata.update({
        "convolution_model": LINEAR_SAME,
        "degradation_psf": True,
        "degradation_image_shape": tuple(int(v) for v in image_shape),
        "degradation_support_limit": (
            tuple(int(v) for v in image_shape)
            if safe_width is None
            else int(safe_width)
        ),
    })
    return PSF(
        fitted.kernel.copy(),
        name=fitted.name,
        raw_kernel=fitted.raw_kernel,
        metadata=metadata,
    )

def kernel_without_refitting(psf: PSF, image_shape: Tuple[int, int]) -> np.ndarray:
    """Return a convolution kernel without re-centering/re-cropping if possible."""
    if not isinstance(psf, PSF):
        psf = PSF(psf)
    kh, kw = psf.kernel.shape
    h, w = int(image_shape[0]), int(image_shape[1])
    if kh <= h and kw <= w:
        return np.asarray(psf.kernel, dtype=np.float64)
    # Emergency path only: a user-supplied PSF is larger than the current image.
    return psf.fitted_to_shape(image_shape, max_width=None).kernel

def zero_outside_psf_rectangle(
    kernel: np.ndarray,
    center: Tuple[int, int],
    support_height: int,
    support_width: int,
) -> np.ndarray:
    """Return a full-size PSF array with zero samples outside a selected rectangle.

    ``center`` uses ``(y, x)`` full-array pixel coordinates.  A rectangle may
    extend outside the array; only its intersection is retained, matching the
    zero-padding convention used later by :meth:`PSF.centered_window`.  The
    retained values are not normalized here because the compact calculation PSF
    is normalized after cropping.
    """
    arr = np.maximum(np.nan_to_num(np.asarray(kernel, dtype=np.float64)), 0.0)
    if arr.ndim != 2:
        raise ValueError("PSF must be a 2D array.")
    height = max(1, int(support_height))
    width = max(1, int(support_width))
    cy, cx = int(center[0]), int(center[1])
    x0 = cx - width // 2
    y0 = cy - height // 2
    x1 = x0 + width
    y1 = y0 + height
    out = np.zeros_like(arr, dtype=np.float64)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0:sy1, sx0:sx1] = arr[sy0:sy1, sx0:sx1]
    return out


def calculation_psf_for_image(
    psf: Optional[PSF],
    image_shape: Tuple[int, int],
    *,
    algorithm_convolution_model: str = LINEAR_SAME,
) -> Optional[PSF]:
    """Return the exact PSF selected for numerical calculations.

    The current PSF object contains the full, possibly thresholded source array
    and Tab-2 metadata describing the selected rectangle and its centre.  This
    helper applies that rectangle exactly once, pads outside-array portions with
    zeros, clips negative samples and normalizes the resulting compact kernel to
    unit sum.  Every reconstruction path and synthetic degradation path should
    use this helper so the GUI preview and the numerical operator cannot diverge.
    """
    if not isinstance(psf, PSF):
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    source = psf.fitted_to_shape((h, w), max_width=None)
    kernel = PSF.normalize_kernel(np.maximum(np.nan_to_num(source.kernel), 0.0))
    metadata = dict(getattr(source, "metadata", {}) or {})
    metadata.setdefault("convolution_model", LINEAR_SAME)
    metadata.setdefault("forward_convolution_model", metadata.get("convolution_model", LINEAR_SAME))
    metadata.update({
        "algorithm_convolution_model": str(algorithm_convolution_model),
        "reconstruction_kernel_source": "current_tab2_calculation_psf",
        "reconstruction_image_shape": (h, w),
        "calculation_psf": True,
        "normalized_sum": float(np.sum(kernel)),
    })
    return PSF(
        kernel,
        name=source.name + "_calculation",
        raw_kernel=source.raw_kernel,
        metadata=metadata,
    )


def reconstruction_psf_for_image(
    known_psf: Optional[PSF],
    degradation_psf: Optional[PSF],
    image_shape: Tuple[int, int],
    *,
    use_exact_degradation_psf: bool = False,
    max_width: Optional[int] = None,
    algorithm_convolution_model: str = LINEAR_SAME,
) -> Optional[PSF]:
    """Backward-compatible wrapper using only the current Tab-2 PSF.

    ``degradation_psf``, ``use_exact_degradation_psf`` and ``max_width`` are
    accepted for older callers and settings files, but deliberately ignored.
    Stored degradation snapshots are never used for reconstruction in v99.
    """
    return calculation_psf_for_image(
        known_psf,
        image_shape,
        algorithm_convolution_model=algorithm_convolution_model,
    )


def degrade_image(
    image: GrayImage,
    psf: PSF,
    noise_sigma: float = 0.01,
    noise_type: str = "Gaussian",
    rng: Optional[np.random.Generator] = None,
) -> GrayImage:
    """Blur an image with a PSF and add one of the supported noise models.

    The function intentionally does not re-fit an already calculation-safe PSF.
    Otherwise the PSF used for degradation could differ from the PSF later used
    for reconstruction.
    """
    data = np.asarray(image.data, dtype=np.float64)
    kernel = kernel_without_refitting(psf, data.shape)
    operator = NumpyLinearSameOperator(kernel, data.shape, dtype=np.float32)
    blurred = operator.forward(data)
    blurred = np.clip(np.nan_to_num(blurred), 0.0, 1.0)

    sigma = max(0.0, float(noise_sigma))
    mode = str(noise_type or "Gaussian").strip().lower()
    rng = np.random.default_rng() if rng is None else rng

    noise_field = np.zeros_like(blurred, dtype=np.float64)
    if sigma <= 0.0:
        noisy = blurred
    elif "poisson" in mode:
        # Interpret sigma as inverse photon-count control: larger sigma means fewer photons.
        photons = max(10.0, 1.0 / max(sigma * sigma, 1e-8))
        noisy = rng.poisson(np.clip(blurred, 0.0, 1.0) * photons) / photons
        noise_field = noisy - blurred
    elif "speckle" in mode or "correlated" in mode:
        noise = rng.normal(0.0, sigma, size=blurred.shape)
        # Spatially correlate the multiplicative noise.
        corr_kernel = PSF.gaussian(size=11, sigma=2.0).kernel
        noise = fftconvolve(noise, corr_kernel, mode="same")
        std = float(noise.std())
        if std > 1e-12:
            noise *= sigma / std
        noisy = blurred * (1.0 + noise)
        # Store the actual additive-equivalent disturbance introduced into the image.
        # For speckle this equals blurred * correlated_noise and its PSD is the one
        # requested by the Wiener-noise option.
        noise_field = noisy - blurred
    else:
        noise_field = rng.normal(0.0, sigma, size=blurred.shape)
        noisy = blurred + noise_field

    noisy_clipped = np.clip(noisy, 0.0, 1.0)
    noise_psd = np.abs(fft2(np.asarray(noise_field, dtype=np.float64))) ** 2
    mean_psd = float(np.mean(noise_psd))
    if mean_psd > 1e-18:
        noise_psd = noise_psd / mean_psd
    else:
        noise_psd = np.ones_like(noise_psd, dtype=np.float64)
    metadata = dict(getattr(image, "metadata", {}) or {})
    metadata.update({
        "blurred_clean": blurred,
        "noise_field": noise_field,
        "noise_psd": noise_psd,
        "noise_type": noise_type,
        "noise_sigma": sigma,
        "convolution_model": LINEAR_SAME,
        "degradation_psf_shape": tuple(int(v) for v in kernel.shape),
        "_preserve_intensity": True,
    })
    return GrayImage(
        noisy_clipped,
        name=image.name + "_degraded",
        metadata=metadata,
    )

def total_variation_norm(data: np.ndarray) -> float:
    """Return the mean isotropic total-variation norm of a grayscale image."""
    arr = np.asarray(data, dtype=np.float64)
    arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)
    if arr.ndim != 2 or min(arr.shape) < 2:
        return float("nan")
    dy = np.diff(arr, axis=0)
    dx = np.diff(arr, axis=1)
    tv_y = np.abs(dy).sum()
    tv_x = np.abs(dx).sum()
    return float((tv_x + tv_y) / max(1, arr.size))

def normalized_total_variation(data: np.ndarray, eps: float = 1e-12) -> float:
    """Return TV normalized by the mean absolute image intensity.

    This quantity is approximately invariant to global intensity scaling and is
    therefore more useful than raw TV when no independent reference image is
    available.
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan")
    scale = float(np.mean(np.abs(arr)))
    tv = total_variation_norm(arr)
    return float(tv / max(scale, eps)) if np.isfinite(tv) else float("nan")


def convolution_boundary_mismatch(
    data: np.ndarray,
    kernel: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compare linear-same and circular-FFT convolution on the same array.

    The value is a diagnostic of boundary sensitivity, not a PSF mismatch
    measure.  It is near zero when the array has a sufficiently wide dark
    frame for the selected kernel and grows when circular wrap-around changes
    the forward result.
    """
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2 or x.size == 0:
        return float("nan")
    linear = NumpyLinearSameOperator(kernel, x.shape, dtype=np.float32).forward(x)
    circular = circular_convolve_numpy(x, kernel, dtype=np.float32)
    numerator = float(np.linalg.norm((linear - circular).ravel()))
    denominator = float(np.linalg.norm(linear.ravel()))
    return numerator / max(denominator, eps)


def relative_reblur_residual(
    estimate: np.ndarray,
    measured: np.ndarray,
    psf: Optional[PSF],
    roi_source: Optional[GrayImage] = None,
    eps: float = 1e-12,
) -> float:
    """Return the relative L2 error between the measurement and reblurred estimate.

    When no PSF is available, the estimate itself is compared with the measured
    image. This fallback is deliberately weak; blind methods normally provide
    an estimated PSF in the reconstruction metadata.
    """
    y = np.asarray(measured, dtype=np.float64)
    x = _match_array_shape(np.asarray(estimate, dtype=np.float64), y.shape)
    if psf is not None:
        kernel = np.asarray(psf.kernel, dtype=np.float64)
        convolution_model = str(getattr(psf, "metadata", {}).get("convolution_model", "linear_same"))
        if convolution_model == CIRCULAR_FFT:
            predicted = circular_convolve_numpy(x, kernel, dtype=np.float32)
        else:
            predicted = NumpyLinearSameOperator(kernel, y.shape, dtype=np.float32).forward(x)
    else:
        predicted = x
    predicted = crop_to_original_region(predicted, roi_source)
    y_roi = crop_to_original_region(y, roi_source)
    numerator = float(np.linalg.norm((predicted - y_roi).ravel()))
    denominator = float(np.linalg.norm(y_roi.ravel()))
    return numerator / max(denominator, eps)


def wiener_gcv_cost(
    measured: np.ndarray,
    psf: Optional[PSF],
    k: float,
    noise_psd: Optional[np.ndarray] = None,
    eps: float = 1e-18,
) -> float:
    """Return the generalized cross-validation cost for Wiener regularization.

    For the circular FFT model the fitted-data operator is diagonal in the
    Fourier domain, with leverage ``q = |H|^2 / (|H|^2 + K N)``.  GCV balances
    the residual energy against the effective number of fitted degrees of
    freedom and therefore does not require a reference image or an arbitrary TV
    weight.  The measured image is mean-centered so that a large DC background
    does not dominate selection of K.  Lower values are better.
    """
    if psf is None:
        return float("nan")
    y = np.asarray(measured, dtype=np.float64)
    if y.ndim != 2 or y.size == 0:
        return float("nan")
    y = np.nan_to_num(y)
    y = y - float(np.mean(y))
    H = psf_to_otf_numpy(psf.kernel, y.shape, dtype=np.float32)
    power = np.abs(H) ** 2
    kval = max(float(k), eps)
    if noise_psd is None:
        N = 1.0
    else:
        N = np.asarray(noise_psd, dtype=np.float64)
        if N.shape != y.shape:
            N = _match_array_shape(N, y.shape)
        N = np.maximum(np.nan_to_num(N, nan=1.0, posinf=1.0, neginf=1.0), 0.0)
        mean_n = float(np.mean(N))
        N = N / mean_n if mean_n > eps else np.ones_like(y, dtype=np.float64)
    denominator = power + kval * N
    leverage = power / np.maximum(denominator, eps)
    residual_transfer = 1.0 - leverage
    Y = fft2(y)
    signal_energy = float(np.mean(np.abs(Y) ** 2))
    residual_energy = float(np.mean(np.abs(residual_transfer * Y) ** 2)) / max(signal_energy, eps)
    residual_dof_fraction = float(np.mean(residual_transfer))
    return float(residual_energy / max(residual_dof_fraction ** 2, eps))


def _prepare_psf_candidate_window(
    source_psf: np.ndarray,
    center: Tuple[int, int],
    support_height: int,
    support_width: int,
    floor_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Threshold, crop and unit-normalize one PSF candidate.

    The operation is intentionally identical to the numerical effect of Tab 2
    after ``Apply``: a constant floor is subtracted, negative samples are set to
    zero, the current rectangular frame is extracted (with zero padding outside
    the source array), and only then is the compact kernel normalized to unit
    sum.  The extra diagnostics are used to reject collapsed, nearly impulsive
    candidates during no-reference optimization.
    """
    p_full = np.maximum(np.nan_to_num(np.asarray(source_psf, dtype=np.float64)), 0.0)
    if p_full.ndim != 2 or p_full.size == 0:
        raise ValueError("PSF must be a non-empty 2D array.")
    peak = float(np.max(p_full))
    if peak <= 1e-18:
        raise ValueError("PSF peak must be positive.")
    height = min(max(1, int(support_height)), p_full.shape[0])
    width = min(max(1, int(support_width)), p_full.shape[1])
    cy = int(np.clip(int(round(center[0])), 0, p_full.shape[0] - 1))
    cx = int(np.clip(int(round(center[1])), 0, p_full.shape[1] - 1))
    floor_fraction = float(np.clip(floor_fraction, 0.0, 1.0))
    cutoff = floor_fraction * peak

    source_crop = PSF.centered_window(p_full, (cy, cx), height, width)
    source_crop = np.maximum(np.nan_to_num(source_crop), 0.0)
    source_crop_sum = float(np.sum(source_crop))

    thresholded = np.maximum(p_full - cutoff, 0.0)
    crop = PSF.centered_window(thresholded, (cy, cx), height, width)
    crop_sum = float(np.sum(crop))
    if crop_sum <= 1e-18:
        raise ValueError("The selected PSF window is empty at this floor.")
    crop = crop / crop_sum
    nonzero_pixels = int(np.count_nonzero(crop))
    effective_pixels = float(1.0 / max(float(np.sum(crop * crop)), 1e-18))
    retained_mass_fraction = crop_sum / max(source_crop_sum, 1e-18)
    return crop, {
        "center": (cy, cx),
        "support_height": int(height),
        "support_width": int(width),
        "floor_fraction": floor_fraction,
        "cutoff": float(cutoff),
        "source_crop_sum": source_crop_sum,
        "crop_sum_before_normalization": crop_sum,
        "retained_mass_fraction": float(retained_mass_fraction),
        "normalized_sum": float(np.sum(crop)),
        "nonzero_pixels": nonzero_pixels,
        "nonzero_fraction": float(nonzero_pixels / max(1, crop.size)),
        "effective_pixels": effective_pixels,
    }


def _psf_floor_background_statistics(
    source_psf: np.ndarray,
    center: Tuple[int, int],
    support_height: int,
    support_width: int,
) -> Dict[str, float]:
    """Estimate the plausible PSF-floor interval from the selected frame edge.

    Jointly estimating a PSF and an object from one blurred image is ill posed.
    In particular, GCV can favour a nearly impulsive PSF because GCV is intended
    to tune regularization for a *fixed* forward operator.  The measured PSF
    itself therefore supplies the admissible floor range.  Median/MAD edge
    statistics make the estimate insensitive to a few bright edge samples.
    """
    p_full = np.maximum(np.nan_to_num(np.asarray(source_psf, dtype=np.float64)), 0.0)
    peak = float(np.max(p_full)) if p_full.size else 0.0
    if p_full.ndim != 2 or peak <= 1e-18:
        return {
            "background_fraction": 0.0,
            "sigma_fraction": 0.0,
            "lower_fraction": 0.0,
            "upper_fraction": 0.02,
        }
    h = min(max(1, int(support_height)), p_full.shape[0])
    w = min(max(1, int(support_width)), p_full.shape[1])
    cy = int(np.clip(int(round(center[0])), 0, p_full.shape[0] - 1))
    cx = int(np.clip(int(round(center[1])), 0, p_full.shape[1] - 1))
    crop = PSF.centered_window(p_full, (cy, cx), h, w)
    if crop.shape[0] == 1 or crop.shape[1] == 1:
        border = crop.ravel()
    else:
        border = np.concatenate((crop[0], crop[-1], crop[1:-1, 0], crop[1:-1, -1]))
    border = np.maximum(np.nan_to_num(border), 0.0)
    if not border.size:
        border = np.zeros(1, dtype=np.float64)
    median = float(np.median(border))
    mad = float(np.median(np.abs(border - median)))
    sigma = 1.4826 * mad
    q99 = float(np.quantile(border, 0.99))
    bg_fraction = float(np.clip(median / peak, 0.0, 1.0))
    sigma_fraction = float(max(0.0, sigma / peak))
    q99_fraction = float(np.clip(q99 / peak, 0.0, 1.0))

    # Automatic thresholding is deliberately conservative.  A value above 25%
    # of the PSF peak is almost never a background estimate and was the source
    # of the collapsed two-pixel kernels observed in v101.
    lower = max(0.0, bg_fraction - 3.0 * sigma_fraction)
    upper = max(0.02, 1.5 * bg_fraction, q99_fraction + 3.0 * sigma_fraction,
                bg_fraction + 6.0 * sigma_fraction)
    upper = float(np.clip(upper, lower + 1e-5, 0.25))
    lower = float(np.clip(lower, 0.0, max(0.0, upper - 1e-5)))
    return {
        "background_fraction": bg_fraction,
        "sigma_fraction": sigma_fraction,
        "q99_fraction": q99_fraction,
        "lower_fraction": lower,
        "upper_fraction": upper,
    }


def optimize_psf_floor_and_wiener_k(
    measured: np.ndarray,
    source_psf: np.ndarray,
    center: Tuple[int, int],
    support_width: int,
    support_height: Optional[int] = None,
    reference: Optional[np.ndarray] = None,
    current_k: float = 1e-2,
    current_floor: float = 0.0,
    max_preview_side: int = 256,
) -> Dict[str, Any]:
    """Tune the PSF floor and Wiener ``K`` without allowing PSF collapse.

    With an independent reference, the two parameters are selected directly by
    reconstruction MSE.  Without a reference, the problem is treated as a
    nested, constrained optimization:

    * the admissible floor interval is estimated from robust edge statistics of
      the currently selected PSF rectangle;
    * for each admissible floor, ``K`` is chosen by ordinary Wiener GCV for that
      *fixed* PSF;
    * floor selection adds a PSF-background prior and rejects candidates that
      collapse to only a few effective pixels.

    This avoids comparing unconstrained GCV values across arbitrarily different
    forward operators, which can otherwise prefer a nearly delta-like PSF.
    """
    y_full = np.maximum(np.nan_to_num(np.asarray(measured, dtype=np.float64)), 0.0)
    p_full = np.maximum(np.nan_to_num(np.asarray(source_psf, dtype=np.float64)), 0.0)
    if y_full.ndim != 2 or p_full.ndim != 2 or y_full.size == 0 or p_full.size == 0:
        raise ValueError("Measured image and PSF must be non-empty 2D arrays.")
    peak = float(np.max(p_full))
    if peak <= 1e-18:
        raise ValueError("PSF peak must be positive.")
    support_height = int(support_width if support_height is None else support_height)
    support_width = int(support_width)

    scale = min(1.0, float(max_preview_side) / float(max(y_full.shape)))
    if scale < 1.0:
        ph = max(16, int(round(y_full.shape[0] * scale)))
        pw = max(16, int(round(y_full.shape[1] * scale)))
        y = GrayImage.resize_array(y_full, width=pw, height=ph)
        ref = GrayImage.resize_array(np.asarray(reference, dtype=np.float64), width=pw, height=ph) if reference is not None else None
    else:
        y = y_full.copy()
        ref = np.asarray(reference, dtype=np.float64).copy() if reference is not None else None
    if ref is not None:
        ref = _match_array_shape(ref, y.shape)
        ref = np.clip(np.nan_to_num(ref), 0.0, 1.0)

    current_floor = float(np.clip(current_floor, 0.0, 0.95))
    current_k = max(1e-12, float(current_k))
    background = _psf_floor_background_statistics(
        p_full, center, support_height, support_width
    )

    if ref is not None:
        # A real reference makes broad floor exploration identifiable.
        automatic = PSF.automatic_support_selection(p_full, peak_fraction=1e-2)
        base_floor = float(np.clip(automatic.get("floor_fraction", 0.0), 0.0, 0.5))
        upper_floor = float(np.clip(max(0.20, 4.0 * current_floor, 4.0 * base_floor), 1e-4, 0.95))
        floors = np.unique(np.concatenate((
            np.array([0.0, current_floor, base_floor, 0.01, 0.02, 0.05, 0.10], dtype=np.float64),
            np.geomspace(1e-6, upper_floor, 17, dtype=np.float64),
            np.linspace(0.0, upper_floor, 17, dtype=np.float64),
        )))
    else:
        lower = float(background["lower_fraction"])
        upper = float(background["upper_fraction"])
        bg = float(background["background_fraction"])
        candidates = [0.0, lower, bg, upper]
        if lower < upper:
            candidates.extend(np.linspace(lower, upper, 19, dtype=np.float64).tolist())
        if lower <= current_floor <= upper:
            candidates.append(current_floor)
        floors = np.unique(np.clip(np.asarray(candidates, dtype=np.float64), lower, upper))

    coarse_ks = np.unique(np.concatenate((
        np.geomspace(1e-12, 1e2, 29, dtype=np.float64),
        np.array([current_k], dtype=np.float64),
    )))
    y32 = np.asarray(y, dtype=np.float32)
    Y = fft2(y32)
    y_centered = y32 - np.float32(np.mean(y32))
    Y_centered = fft2(y_centered)
    centered_energy = float(np.mean(np.abs(Y_centered) ** 2))
    criterion_name = "MSE" if ref is not None else "Constrained PSF background + conditional Wiener GCV"
    best = {"cost": float("inf"), "floor_fraction": current_floor, "K": current_k}
    evaluations = 0
    initial_cost = float("nan")
    best_candidate_meta: Dict[str, Any] = {}
    best_details: Dict[str, float] = {}

    def candidate_otf(floor_fraction: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        crop, meta = _prepare_psf_candidate_window(
            p_full, center, support_height, support_width, floor_fraction
        )
        if scale < 1.0:
            scaled_h = min(max(1, int(round(crop.shape[0] * scale))), y.shape[0])
            scaled_w = min(max(1, int(round(crop.shape[1] * scale))), y.shape[1])
            crop = GrayImage.resize_array(crop, width=scaled_w, height=scaled_h)
            crop = PSF.normalize_kernel(crop)
            meta = dict(meta)
            meta["preview_support"] = tuple(int(v) for v in crop.shape)
            meta["preview_normalized_sum"] = float(np.sum(crop))
            meta["preview_effective_pixels"] = float(1.0 / max(float(np.sum(crop * crop)), 1e-18))
        return psf_to_otf_numpy(crop, y.shape, dtype=np.float32), meta

    def conditional_gcv(H: np.ndarray, kval: float) -> float:
        kval = max(1e-12, float(kval))
        power = np.abs(H) ** 2
        denominator = np.maximum(power + np.float32(kval), 1e-18)
        residual_transfer = 1.0 - power / denominator
        residual_energy = float(np.mean(np.abs(residual_transfer * Y_centered) ** 2)) / max(centered_energy, 1e-18)
        dof_fraction = float(np.mean(residual_transfer))
        return float(residual_energy / max(dof_fraction * dof_fraction, 1e-18))

    def reconstruction_and_whiteness(H: np.ndarray, kval: float) -> Tuple[np.ndarray, float]:
        power = np.abs(H) ** 2
        denominator = np.maximum(power + np.float32(max(1e-12, kval)), 1e-18)
        estimate_arr = np.real(ifft2(np.conj(H) * Y / denominator)).astype(np.float32, copy=False)
        estimate_arr = np.clip(np.nan_to_num(estimate_arr), 0.0, 1.0)
        predicted = np.real(ifft2(H * fft2(estimate_arr))).astype(np.float32, copy=False)
        residual = np.nan_to_num(predicted - y32)
        centered = residual - np.float32(np.mean(residual))
        variance = float(np.mean(centered * centered))
        correlations: List[float] = []
        if variance > 1e-18:
            for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)):
                y0a, y1a = max(0, dy), centered.shape[0] + min(0, dy)
                x0a, x1a = max(0, dx), centered.shape[1] + min(0, dx)
                y0b, y1b = max(0, -dy), centered.shape[0] - max(0, dy)
                x0b, x1b = max(0, -dx), centered.shape[1] - max(0, dx)
                a = centered[y0a:y1a, x0a:x1a]
                b = centered[y0b:y1b, x0b:x1b]
                if a.size and a.shape == b.shape:
                    rho = float(np.mean(a * b) / max(variance, 1e-18))
                    correlations.append(rho * rho)
        return estimate_arr, float(np.mean(correlations)) if correlations else 0.0

    def floor_selection_cost(H: np.ndarray, meta: Dict[str, Any], floor_fraction: float, kval: float) -> Tuple[float, Dict[str, float]]:
        if ref is not None:
            estimate_arr, whiteness = reconstruction_and_whiteness(H, kval)
            mse = float(np.mean((estimate_arr - ref) ** 2))
            return mse, {"MSE": mse, "residual_whiteness": whiteness}

        # GCV selects K conditionally on one fixed H.  A separate PSF-derived
        # prior chooses between plausible floors and prevents support collapse.
        gcv = conditional_gcv(H, kval)
        _, whiteness = reconstruction_and_whiteness(H, kval)
        bg = float(background["background_fraction"])
        sigma = max(float(background["sigma_fraction"]), 0.003, 0.15 * max(bg, 0.01))
        z = (float(floor_fraction) - bg) / sigma
        prior = 0.20 * z * z
        retained = float(meta.get("retained_mass_fraction", 1.0))
        effective = float(meta.get("preview_effective_pixels", meta.get("effective_pixels", 1.0)))
        min_effective = max(2.5, 0.0025 * float(max(1, support_height * support_width)))
        collapse = 0.0
        if retained < 0.10:
            collapse += 4.0 * ((0.10 - retained) / 0.10) ** 2
        if effective < min_effective:
            collapse += 4.0 * ((min_effective - effective) / min_effective) ** 2
        cost = float(np.log(max(gcv, 1e-18)) + 0.10 * whiteness + prior + collapse)
        return cost, {
            "conditional_gcv": float(gcv),
            "residual_whiteness": float(whiteness),
            "psf_background_prior": float(prior),
            "psf_collapse_penalty": float(collapse),
            "retained_mass_fraction": retained,
            "effective_pixels": effective,
        }

    def best_k_for_floor(H: np.ndarray, meta: Dict[str, Any], floor_fraction: float) -> Tuple[float, float, Dict[str, float], int]:
        local_best_k = current_k
        local_best_cost = float("inf")
        local_details: Dict[str, float] = {}
        tested = 0
        for kval in coarse_ks:
            if ref is None:
                # For a fixed PSF, choose K by GCV only.
                k_cost = conditional_gcv(H, float(kval))
            else:
                k_cost, _ = floor_selection_cost(H, meta, floor_fraction, float(kval))
            tested += 1
            if np.isfinite(k_cost) and k_cost < local_best_cost:
                local_best_cost = float(k_cost)
                local_best_k = float(kval)
        fine_ks = np.geomspace(max(1e-12, local_best_k / 10.0), min(1e4, local_best_k * 10.0), 15)
        for kval in fine_ks:
            if ref is None:
                k_cost = conditional_gcv(H, float(kval))
            else:
                k_cost, _ = floor_selection_cost(H, meta, floor_fraction, float(kval))
            tested += 1
            if np.isfinite(k_cost) and k_cost < local_best_cost:
                local_best_cost = float(k_cost)
                local_best_k = float(kval)
        selection_cost, local_details = floor_selection_cost(H, meta, floor_fraction, local_best_k)
        return local_best_k, selection_cost, local_details, tested

    # Initial diagnostics use the actual current pair, even when the no-reference
    # automatic search later restricts the floor interval.
    try:
        H0, meta0 = candidate_otf(current_floor)
        initial_cost, initial_details = floor_selection_cost(H0, meta0, current_floor, current_k)
    except ValueError:
        initial_details = {}

    for floor_fraction in floors:
        try:
            H, candidate_meta = candidate_otf(float(floor_fraction))
        except ValueError:
            continue
        kval, cost, details, tested = best_k_for_floor(H, candidate_meta, float(floor_fraction))
        evaluations += tested
        if np.isfinite(cost) and cost < float(best["cost"]):
            best = {"cost": float(cost), "floor_fraction": float(floor_fraction), "K": float(kval)}
            best_candidate_meta = dict(candidate_meta)
            best_details = dict(details)

    if not np.isfinite(float(best["cost"])):
        raise ValueError("No valid PSF-floor/Wiener-K candidate was found.")

    best.update({
        "criterion": criterion_name,
        "evaluations": int(evaluations),
        "preview_shape": tuple(int(v) for v in y.shape),
        "support_height": int(support_height),
        "support_width": int(support_width),
        "center": tuple(int(v) for v in center),
        "initial_floor_fraction": float(current_floor),
        "initial_K": float(current_k),
        "initial_cost": float(initial_cost),
        "cost_improvement": float(initial_cost - float(best["cost"])) if np.isfinite(initial_cost) else float("nan"),
        "candidate_psf": best_candidate_meta,
        "criterion_components": best_details,
        "initial_criterion_components": initial_details,
        "psf_background": background,
        "floor_search_bounds": (
            float(np.min(floors)) if len(floors) else float(current_floor),
            float(np.max(floors)) if len(floors) else float(current_floor),
        ),
        "criterion_details": (
            "MSE against the independent reference"
            if ref is not None else
            "K is selected by GCV for each fixed PSF; floor is selected only within the robust PSF-background interval with collapse prevention"
        ),
    })
    return best



def auto_tunable_parameter_names(
    algorithm_name: str,
    active_names: List[str],
    initial_params: Dict[str, Any],
) -> List[str]:
    """Return parameters that Auto may change for one frozen initial state.

    Feature-enabling controls are intentionally frozen for the complete Auto
    run.  Consequently, a disabled Wiener initializer, denoiser or optional TV
    step cannot be enabled temporarily and expose its dependent parameters to a
    later coordinate-search pass.  This also guarantees that values hidden by a
    disabled option remain byte-for-byte unchanged after Auto.
    """
    allowed = set(str(name) for name in active_names)
    alg = str(algorithm_name)

    # Auto may tune other independent booleans, but it must not change whether
    # these optional processing stages are present in the pipeline.
    activation_controls = {
        "begin_with_wiener",
        "use_tv_preconditioning",
        "neural_denoiser_mode",
        "wiener_use_noise_psd",
        "rosen_relax_to_one",
        "blind_use_known_psf_init",
    }
    allowed.difference_update(activation_controls)

    direct_wiener_k = {
        "Wiener", "Torch batch Wiener",
        "Richardson-Lucy-Wiener", "Torch batch Richardson-Lucy-Wiener",
        "Landweber Wiener-preconditioned",
    }
    wiener_stage_active = alg in direct_wiener_k or bool(initial_params.get("begin_with_wiener", False))
    if not wiener_stage_active:
        allowed.discard("K")
        allowed.discard("wiener_use_noise_psd")

    denoiser_active = str(initial_params.get("neural_denoiser_mode", "Off")) != "Off"
    if not denoiser_active:
        allowed.difference_update({
            "denoiser_type", "neural_denoiser_strength", "neural_denoiser_weights"
        })

    intrinsic_tv = alg in {
        "PyTorch Adam TV-MAP", "PyTorch Blind Adam TV-MAP",
    }
    denoiser_tv = denoiser_active and str(initial_params.get("denoiser_type", "")) == "TV only"
    optional_tv_active = bool(initial_params.get("use_tv_preconditioning", False))
    if not (intrinsic_tv or denoiser_tv or optional_tv_active):
        allowed.discard("tv_weight")
        allowed.discard("tv_iterations")

    if not bool(initial_params.get("rosen_relax_to_one", False)):
        allowed.discard("rosen_relax_factor")

    return [name for name in active_names if str(name) in allowed]

def residual_whiteness_cost(
    estimate: np.ndarray,
    measured: np.ndarray,
    psf: Optional[PSF],
    roi_source: Optional[GrayImage] = None,
    eps: float = 1e-12,
) -> float:
    """Measure short-range spatial correlation remaining in the reblur residual.

    A residual that resembles independent noise has small normalized
    autocorrelation away from zero lag.  Strongly over-smoothed reconstructions
    leave edges and broad structures in the residual and therefore receive a
    larger penalty.  Lower values are better.
    """
    y = np.asarray(measured, dtype=np.float64)
    x = _match_array_shape(np.asarray(estimate, dtype=np.float64), y.shape)
    if psf is not None:
        kernel = np.asarray(psf.kernel, dtype=np.float64)
        convolution_model = str(getattr(psf, "metadata", {}).get("convolution_model", "linear_same"))
        if convolution_model == CIRCULAR_FFT:
            predicted = circular_convolve_numpy(x, kernel, dtype=np.float32)
        else:
            predicted = NumpyLinearSameOperator(kernel, y.shape, dtype=np.float32).forward(x)
    else:
        predicted = x
    residual = crop_to_original_region(predicted - y, roi_source)
    residual = np.asarray(residual, dtype=np.float64)
    residual = np.nan_to_num(residual)
    residual = residual - float(np.mean(residual))
    variance = float(np.mean(residual ** 2))
    if variance <= eps or min(residual.shape) < 3:
        return 0.0
    correlations: List[float] = []
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)):
        y0a, y1a = max(0, dy), residual.shape[0] + min(0, dy)
        x0a, x1a = max(0, dx), residual.shape[1] + min(0, dx)
        y0b, y1b = max(0, -dy), residual.shape[0] - max(0, dy)
        x0b, x1b = max(0, -dx), residual.shape[1] - max(0, dx)
        a = residual[y0a:y1a, x0a:x1a]
        b = residual[y0b:y1b, x0b:x1b]
        if a.size and b.size and a.shape == b.shape:
            rho = float(np.mean(a * b) / max(variance, eps))
            correlations.append(rho * rho)
    return float(np.mean(correlations)) if correlations else 0.0


def no_reference_quality_cost(
    estimate: np.ndarray,
    measured: Optional[np.ndarray],
    psf: Optional[PSF],
    roi_source: Optional[GrayImage] = None,
    tv_weight: float = 0.005,
    intensity_weight: float = 0.01,
    whiteness_weight: float = 0.25,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Compute a simple no-reference reconstruction cost.

    The cost combines measurement consistency, lightly weighted normalized TV,
    intensity preservation and short-range residual whiteness. Lower is better.
    Plain Wiener Auto uses generalized cross-validation instead of this generic
    image-domain criterion.
    """
    x_full = np.asarray(estimate, dtype=np.float64)
    x_roi = crop_to_original_region(x_full, roi_source)
    ntv = normalized_total_variation(x_roi, eps=eps)
    residual = float("nan")
    intensity_error = 0.0
    whiteness = float("nan")
    if measured is not None:
        y_full = np.asarray(measured, dtype=np.float64)
        y_roi = crop_to_original_region(y_full, roi_source)
        residual = relative_reblur_residual(x_full, y_full, psf, roi_source=roi_source, eps=eps)
        whiteness = residual_whiteness_cost(x_full, y_full, psf, roi_source=roi_source, eps=eps)
        mean_y = float(np.mean(np.abs(y_roi)))
        mean_x = float(np.mean(np.abs(x_roi)))
        intensity_error = abs(mean_x - mean_y) / max(mean_y, eps)
    whiteness_term = float(whiteness_weight) * whiteness if np.isfinite(whiteness) else 0.0
    if np.isfinite(residual):
        cost = residual + float(tv_weight) * ntv + float(intensity_weight) * intensity_error + whiteness_term
    else:
        cost = float(tv_weight) * ntv + float(intensity_weight) * intensity_error + whiteness_term
    return {
        "NTV": float(ntv),
        "RELATIVE_REBLUR_RESIDUAL": float(residual),
        "RELATIVE_INTENSITY_ERROR": float(intensity_error),
        "RESIDUAL_WHITENESS": float(whiteness),
        "NO_REFERENCE_COST": float(cost),
    }


def is_measured_input_pair(reference: Optional[GrayImage], degraded: Optional[GrayImage]) -> bool:
    """True when the loaded image is also used as the measured/degraded input.

    In this mode there is no independent ground-truth reference, so PSNR/SSIM
    against the loaded image would be misleading and must not be reported.
    """
    if reference is None or degraded is None:
        return False
    if not bool(degraded.metadata.get("measured_input", False)):
        return False
    try:
        return reference.data.shape == degraded.data.shape and np.allclose(reference.data, degraded.data)
    except Exception:
        return bool(degraded.metadata.get("measured_input", False))

def reference_metrics_available(state: Optional[Dict[str, Any]]) -> bool:
    """Return True only when an independent ground-truth reference is available."""
    if not state:
        return False
    if state.get("reference_available") is False:
        return False
    reference = state.get("image")
    if reference is None:
        return False
    return not is_measured_input_pair(reference, state.get("degraded"))

def original_region_slices(image: Optional[GrayImage], target_shape: Optional[Tuple[int, int]] = None) -> Tuple[slice, slice]:
    """Return slices of the original non-padded image region.

    Images created with ``from_array_with_zero_frame`` store an explicit
    ``content_roi``. Older settings/files may only contain ``zero_padding`` and
    ``inner_size``; those values are used as a backward-compatible fallback.
    """
    if target_shape is None:
        if image is None:
            return slice(None), slice(None)
        target_shape = tuple(np.asarray(image.data).shape)
    h, w = int(target_shape[0]), int(target_shape[1])
    if image is None or not isinstance(getattr(image, "metadata", None), dict):
        return slice(0, h), slice(0, w)
    meta = image.metadata
    roi = meta.get("content_roi")
    if isinstance(roi, (tuple, list)) and len(roi) == 4:
        y0, y1, x0, x1 = [int(v) for v in roi]
    else:
        inner = meta.get("inner_size")
        pad = int(max(0, meta.get("zero_padding", 0) or 0))
        if isinstance(inner, (tuple, list)) and len(inner) == 2:
            ih, iw = int(inner[0]), int(inner[1])
            y0 = max(0, (h - ih) // 2)
            x0 = max(0, (w - iw) // 2)
            y1, x1 = y0 + ih, x0 + iw
        elif pad > 0:
            y0, y1, x0, x1 = pad, h - pad, pad, w - pad
        else:
            return slice(0, h), slice(0, w)
    y0, x0 = max(0, min(h, y0)), max(0, min(w, x0))
    y1, x1 = max(y0 + 1, min(h, y1)), max(x0 + 1, min(w, x1))
    return slice(y0, y1), slice(x0, x1)

def crop_to_original_region(data: np.ndarray, roi_source: Optional[GrayImage]) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim < 2:
        return arr
    ys, xs = original_region_slices(roi_source, arr.shape[-2:])
    return arr[..., ys, xs]

def compute_metrics(
    reference: Optional[GrayImage],
    estimate: Optional[GrayImage],
    allow_reference_metrics: bool = True,
    roi_source: Optional[GrayImage] = None,
    measured: Optional[GrayImage] = None,
    psf: Optional[PSF] = None,
) -> Dict[str, float]:
    """Compute metrics inside the original, non-padded image region.

    PSNR and SSIM are reported only when an independent reference is available.
    Without a reference, a generic no-reference cost combines relative reblur
    residual, lightly weighted normalized TV, intensity preservation and residual
    whiteness. Plain Wiener Auto uses a separate Fourier-domain GCV criterion.
    """
    metrics: Dict[str, float] = {}
    if estimate is None:
        return metrics
    roi_image = reference if reference is not None else (roi_source if roi_source is not None else estimate)
    est_raw = crop_to_original_region(np.asarray(estimate.data, dtype=np.float64), roi_image)
    metrics["TV"] = total_variation_norm(est_raw)
    metrics["NTV"] = normalized_total_variation(est_raw)

    measured_array = np.asarray(measured.data, dtype=np.float64) if measured is not None else None
    metrics.update(no_reference_quality_cost(
        np.asarray(estimate.data, dtype=np.float64),
        measured_array,
        psf,
        roi_source=roi_source if roi_source is not None else roi_image,
    ))

    if reference is None or not allow_reference_metrics:
        return metrics

    ref_full = np.asarray(reference.data, dtype=np.float64)
    est_full = _match_array_shape(np.asarray(estimate.data, dtype=np.float64), ref_full.shape)
    ref = crop_to_original_region(ref_full, reference)
    est = crop_to_original_region(est_full, reference)
    ref = np.clip(np.nan_to_num(ref), 0.0, 1.0)
    est = np.clip(np.nan_to_num(est), 0.0, 1.0)

    try:
        psnr = float(peak_signal_noise_ratio(ref, est, data_range=1.0))
    except Exception:
        psnr = float("nan")
    try:
        min_dim = min(ref.shape)
        if min_dim < 3:
            ssim = float("nan")
        else:
            win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
            win = max(3, win)
            ssim = float(structural_similarity(ref, est, data_range=1.0, win_size=win))
    except Exception:
        ssim = float("nan")
    metrics["PSNR"] = psnr
    metrics["SSIM"] = ssim
    return metrics


def _batch_structural_similarity(reference: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of the default scikit-image grayscale SSIM.

    The first axis of ``estimates`` is the iteration/batch axis and is never
    mixed by the uniform filters.  This avoids a Python call to
    ``structural_similarity`` for every stored iteration.
    """
    ref = np.clip(np.nan_to_num(np.asarray(reference, dtype=np.float64)), 0.0, 1.0)
    est = np.clip(np.nan_to_num(np.asarray(estimates, dtype=np.float64)), 0.0, 1.0)
    if est.ndim != 3 or ref.ndim != 2 or tuple(est.shape[-2:]) != tuple(ref.shape):
        raise ValueError("Batch SSIM expects estimates [N,H,W] and one matching [H,W] reference.")
    min_dim = min(ref.shape)
    if min_dim < 3:
        return np.full(est.shape[0], np.nan, dtype=np.float64)
    win = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    win = max(3, int(win))
    n_pixels = float(win ** 2)
    covariance_normalization = n_pixels / max(n_pixels - 1.0, 1.0)

    ux = uniform_filter(ref, size=win)
    uxx = uniform_filter(ref * ref, size=win)
    uy = uniform_filter(est, size=(1, win, win))
    uyy = uniform_filter(est * est, size=(1, win, win))
    uxy = uniform_filter(est * ref[None, ...], size=(1, win, win))

    vx = covariance_normalization * (uxx - ux * ux)
    vy = covariance_normalization * (uyy - uy * uy)
    vxy = covariance_normalization * (uxy - ux[None, ...] * uy)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * ux[None, ...] * uy + c1) * (2.0 * vxy + c2)
    denominator = (ux[None, ...] ** 2 + uy ** 2 + c1) * (vx[None, ...] + vy + c2)
    ssim_map = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=np.abs(denominator) > 1e-18,
    )
    pad = (win - 1) // 2
    if pad > 0 and 2 * pad < min(ref.shape):
        ssim_map = ssim_map[:, pad:-pad, pad:-pad]
    return np.mean(ssim_map, axis=(-2, -1), dtype=np.float64)


def compute_metrics_batch(
    reference: Optional[GrayImage],
    estimates: List[GrayImage],
    allow_reference_metrics: bool = True,
    roi_source: Optional[GrayImage] = None,
    measured: Optional[GrayImage] = None,
    psfs: Optional[List[Optional[PSF]]] = None,
    prefer_cuda: bool = True,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, float]]:
    """Compute criteria for all stored iterations in Torch batches.

    TV, NTV, PSNR, measurement consistency, intensity preservation and
    residual whiteness are evaluated simultaneously for many iterations.
    Linear and circular reblurring also accept a different PSF for each frame,
    provided kernels with equal shapes are grouped into the same FFT batch.
    SSIM is evaluated by vectorized SciPy uniform filters.  CUDA float32 is used
    when available; otherwise the same batched path runs on the CPU.

    Very large histories are split only when an all-at-once tensor would use an
    unsafe amount of memory.  ``diagnostics`` is populated with the selected
    device, batch size and number of batches.
    """
    frames = [frame for frame in estimates if frame is not None]
    if not frames:
        if diagnostics is not None:
            diagnostics.update({"device": "none", "batch_size": 0, "batches": 0})
        return []
    if len(frames) != len(estimates):
        # Preserve ordering semantics instead of silently dropping None items.
        return [
            compute_metrics(reference, frame, allow_reference_metrics, roi_source, measured, None)
            if frame is not None else {}
            for frame in estimates
        ]

    shape = tuple(int(v) for v in np.asarray(frames[0].data).shape)
    if len(shape) != 2 or any(tuple(np.asarray(frame.data).shape) != shape for frame in frames):
        if diagnostics is not None:
            diagnostics.update({"device": "scalar fallback", "batch_size": 1, "batches": len(frames), "reason": "mixed shapes"})
        return [
            compute_metrics(
                reference, frame, allow_reference_metrics=allow_reference_metrics,
                roi_source=roi_source, measured=measured,
                psf=(psfs[index] if psfs is not None and index < len(psfs) else None),
            )
            for index, frame in enumerate(frames)
        ]

    if psfs is None:
        psf_items: List[Optional[PSF]] = [None] * len(frames)
    else:
        if len(psfs) != len(frames):
            raise ValueError("The PSF list must have the same length as the estimate history.")
        psf_items = list(psfs)

    if not TORCH_AVAILABLE:
        if diagnostics is not None:
            diagnostics.update({"device": "scalar NumPy", "batch_size": 1, "batches": len(frames), "reason": "PyTorch unavailable"})
        return [
            compute_metrics(
                reference, frame, allow_reference_metrics=allow_reference_metrics,
                roi_source=roi_source, measured=measured, psf=psf_items[index],
            )
            for index, frame in enumerate(frames)
        ]

    h, w = shape
    largest_kh = max([int(psf.kernel.shape[0]) for psf in psf_items if isinstance(psf, PSF)] or [1])
    largest_kw = max([int(psf.kernel.shape[1]) for psf in psf_items if isinstance(psf, PSF)] or [1])
    full_pixels = (h + largest_kh - 1) * (w + largest_kw - 1)

    def _safe_batch_size(device: "torch.device") -> int:
        # FFTs and residual metrics coexist temporarily.  The factor is a
        # conservative upper bound rather than the raw image-stack size.
        bytes_per_frame = max(1, 4 * (12 * h * w + 10 * full_pixels))
        if device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                budget = int(0.30 * free_bytes)
            except Exception:
                budget = 512 * 1024 ** 2
        else:
            budget = 768 * 1024 ** 2
        return max(1, min(len(frames), int(max(1, budget // bytes_per_frame))))

    reference_array = None
    if reference is not None and allow_reference_metrics:
        reference_array = _match_array_shape(np.asarray(reference.data, dtype=np.float32), shape)
    measured_array = None
    if measured is not None:
        measured_array = _match_array_shape(np.asarray(measured.data, dtype=np.float32), shape)

    tv_roi_source = reference if reference_array is not None else (roi_source if roi_source is not None else frames[0])
    tv_ys, tv_xs = original_region_slices(tv_roi_source, shape)
    quality_roi_source = roi_source if roi_source is not None else tv_roi_source
    q_ys, q_xs = original_region_slices(quality_roi_source, shape)
    if reference_array is not None:
        ref_ys, ref_xs = original_region_slices(reference, shape)
        reference_roi = np.clip(np.nan_to_num(reference_array[ref_ys, ref_xs]), 0.0, 1.0)
    else:
        ref_ys = ref_xs = None
        reference_roi = None

    def _run_on_device(device: "torch.device") -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
        batch_size = _safe_batch_size(device)
        output: List[Dict[str, float]] = []
        batch_count = 0
        psf_group_count = 0
        eps = 1e-12
        for start in range(0, len(frames), batch_size):
            stop = min(len(frames), start + batch_size)
            batch_count += 1
            batch_frames = frames[start:stop]
            batch_psfs = psf_items[start:stop]
            x_np = np.stack([np.asarray(frame.data, dtype=np.float32) for frame in batch_frames], axis=0)
            x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
            x = torch.nan_to_num(x)

            tv_roi = torch.clamp(x[..., tv_ys, tv_xs], 0.0, 1.0)
            if min(int(tv_roi.shape[-2]), int(tv_roi.shape[-1])) < 2:
                tv = torch.full((x.shape[0],), float("nan"), device=device)
            else:
                dy = torch.abs(tv_roi[..., 1:, :] - tv_roi[..., :-1, :]).sum(dim=(-2, -1))
                dx = torch.abs(tv_roi[..., :, 1:] - tv_roi[..., :, :-1]).sum(dim=(-2, -1))
                tv = (dx + dy) / float(max(1, int(tv_roi.shape[-2] * tv_roi.shape[-1])))
            quality_clipped = torch.clamp(x[..., q_ys, q_xs], 0.0, 1.0)
            if min(int(quality_clipped.shape[-2]), int(quality_clipped.shape[-1])) < 2:
                quality_tv = torch.full((x.shape[0],), float("nan"), device=device)
            else:
                quality_dy = torch.abs(quality_clipped[..., 1:, :] - quality_clipped[..., :-1, :]).sum(dim=(-2, -1))
                quality_dx = torch.abs(quality_clipped[..., :, 1:] - quality_clipped[..., :, :-1]).sum(dim=(-2, -1))
                quality_tv = (quality_dx + quality_dy) / float(
                    max(1, int(quality_clipped.shape[-2] * quality_clipped.shape[-1]))
                )
            quality_scale = torch.mean(torch.abs(quality_clipped), dim=(-2, -1))
            ntv = quality_tv / torch.clamp(quality_scale, min=eps)

            if measured_array is not None:
                predicted = torch.empty_like(x)
                groups: Dict[Tuple[Any, ...], List[int]] = {}
                for local_index, psf in enumerate(batch_psfs):
                    if not isinstance(psf, PSF):
                        key = ("none",)
                    else:
                        model = str((getattr(psf, "metadata", {}) or {}).get("convolution_model", LINEAR_SAME))
                        key = (model, tuple(int(v) for v in psf.kernel.shape))
                    groups.setdefault(key, []).append(local_index)
                psf_group_count += len(groups)
                for key, local_indices in groups.items():
                    index_tensor = torch.as_tensor(local_indices, dtype=torch.long, device=device)
                    sub_x = x.index_select(0, index_tensor)
                    if key[0] == "none":
                        sub_predicted = sub_x
                    else:
                        model = str(key[0])
                        kh, kw = [int(v) for v in key[1]]
                        kernels_np = np.stack([
                            np.asarray(batch_psfs[index].kernel, dtype=np.float32)
                            for index in local_indices
                        ], axis=0)
                        kernels = torch.as_tensor(kernels_np, dtype=torch.float32, device=device)
                        if model == CIRCULAR_FFT:
                            if kh > h or kw > w:
                                raise ValueError(f"Circular PSF shape {(kh, kw)} exceeds image shape {(h, w)}.")
                            padded = torch.zeros((len(local_indices), h, w), dtype=torch.float32, device=device)
                            padded[:, :kh, :kw] = kernels
                            padded = torch.roll(padded, shifts=-(kh // 2), dims=-2)
                            padded = torch.roll(padded, shifts=-(kw // 2), dims=-1)
                            otf = torch.fft.fft2(padded)
                            sub_predicted = torch.real(torch.fft.ifft2(torch.fft.fft2(sub_x) * otf))
                        else:
                            full_shape = (h + kh - 1, w + kw - 1)
                            image_spectrum = torch.fft.rfft2(sub_x, s=full_shape)
                            kernel_spectrum = torch.fft.rfft2(kernels, s=full_shape)
                            full = torch.fft.irfft2(image_spectrum * kernel_spectrum, s=full_shape)
                            sy, sx = (kh - 1) // 2, (kw - 1) // 2
                            sub_predicted = full[..., sy:sy + h, sx:sx + w]
                    predicted.index_copy_(0, index_tensor, sub_predicted)

                y = torch.as_tensor(measured_array, dtype=torch.float32, device=device)
                residual_roi = (predicted - y)[..., q_ys, q_xs]
                y_roi = y[q_ys, q_xs]
                residual_norm = torch.linalg.vector_norm(residual_roi, dim=(-2, -1))
                y_norm = torch.linalg.vector_norm(y_roi)
                relative_residual = residual_norm / torch.clamp(y_norm, min=eps)
                x_quality_roi = x[..., q_ys, q_xs]
                mean_y = torch.mean(torch.abs(y_roi))
                mean_x = torch.mean(torch.abs(x_quality_roi), dim=(-2, -1))
                intensity_error = torch.abs(mean_x - mean_y) / torch.clamp(mean_y, min=eps)

                centered = residual_roi - torch.mean(residual_roi, dim=(-2, -1), keepdim=True)
                variance = torch.mean(centered * centered, dim=(-2, -1))
                whiteness_sum = torch.zeros_like(variance)
                correlation_count = 0
                for dy_shift, dx_shift in ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)):
                    y0a, y1a = max(0, dy_shift), int(centered.shape[-2]) + min(0, dy_shift)
                    x0a, x1a = max(0, dx_shift), int(centered.shape[-1]) + min(0, dx_shift)
                    y0b, y1b = max(0, -dy_shift), int(centered.shape[-2]) - max(0, dy_shift)
                    x0b, x1b = max(0, -dx_shift), int(centered.shape[-1]) - max(0, dx_shift)
                    a = centered[..., y0a:y1a, x0a:x1a]
                    b = centered[..., y0b:y1b, x0b:x1b]
                    if a.numel() and tuple(a.shape) == tuple(b.shape):
                        rho = torch.mean(a * b, dim=(-2, -1)) / torch.clamp(variance, min=eps)
                        whiteness_sum = whiteness_sum + rho * rho
                        correlation_count += 1
                whiteness = whiteness_sum / float(max(1, correlation_count))
                if min(int(residual_roi.shape[-2]), int(residual_roi.shape[-1])) < 3:
                    whiteness = torch.zeros_like(whiteness)
                else:
                    whiteness = torch.where(variance <= eps, torch.zeros_like(whiteness), whiteness)
                no_reference_cost = relative_residual + 0.005 * ntv + 0.01 * intensity_error + 0.25 * whiteness
            else:
                relative_residual = torch.full_like(ntv, float("nan"))
                intensity_error = torch.zeros_like(ntv)
                whiteness = torch.full_like(ntv, float("nan"))
                no_reference_cost = 0.005 * ntv

            if reference_roi is not None:
                est_reference_roi = torch.clamp(x[..., ref_ys, ref_xs], 0.0, 1.0)
                ref_tensor = torch.as_tensor(reference_roi, dtype=torch.float32, device=device)
                mse = torch.mean((est_reference_roi - ref_tensor) ** 2, dim=(-2, -1))
                psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-30))
                psnr = torch.where(mse <= 0.0, torch.full_like(psnr, float("inf")), psnr)
                est_reference_np = est_reference_roi.detach().cpu().numpy()
                ssim = _batch_structural_similarity(reference_roi, est_reference_np)
                psnr_np = psnr.detach().cpu().numpy().astype(np.float64)
            else:
                psnr_np = np.full(stop - start, np.nan, dtype=np.float64)
                ssim = np.full(stop - start, np.nan, dtype=np.float64)

            tv_np = tv.detach().cpu().numpy().astype(np.float64)
            ntv_np = ntv.detach().cpu().numpy().astype(np.float64)
            residual_np = relative_residual.detach().cpu().numpy().astype(np.float64)
            intensity_np = intensity_error.detach().cpu().numpy().astype(np.float64)
            whiteness_np = whiteness.detach().cpu().numpy().astype(np.float64)
            cost_np = no_reference_cost.detach().cpu().numpy().astype(np.float64)
            for local_index in range(stop - start):
                item = {
                    "TV": float(tv_np[local_index]),
                    "NTV": float(ntv_np[local_index]),
                    "RELATIVE_REBLUR_RESIDUAL": float(residual_np[local_index]),
                    "RELATIVE_INTENSITY_ERROR": float(intensity_np[local_index]),
                    "RESIDUAL_WHITENESS": float(whiteness_np[local_index]),
                    "NO_REFERENCE_COST": float(cost_np[local_index]),
                }
                if reference_roi is not None:
                    item["PSNR"] = float(psnr_np[local_index])
                    item["SSIM"] = float(ssim[local_index])
                output.append(item)
            del x, tv_roi
        return output, {
            "device": str(device),
            "batch_size": int(batch_size),
            "batches": int(batch_count),
            "frames": int(len(frames)),
            "psf_groups": int(psf_group_count),
            "dtype": "float32",
        }

    primary_device = torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")
    try:
        result, info = _run_on_device(primary_device)
    except Exception as exc:
        if primary_device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                result, info = _run_on_device(torch.device("cpu"))
                info["cuda_fallback"] = f"{type(exc).__name__}: {exc}"
            except Exception as cpu_exc:
                if diagnostics is not None:
                    diagnostics.update({
                        "device": "scalar fallback",
                        "batch_size": 1,
                        "batches": len(frames),
                        "reason": f"CUDA: {exc}; CPU batch: {cpu_exc}",
                    })
                return [
                    compute_metrics(
                        reference, frame, allow_reference_metrics=allow_reference_metrics,
                        roi_source=roi_source, measured=measured, psf=psf_items[index],
                    )
                    for index, frame in enumerate(frames)
                ]
        else:
            if diagnostics is not None:
                diagnostics.update({"device": "scalar fallback", "batch_size": 1, "batches": len(frames), "reason": str(exc)})
            return [
                compute_metrics(
                    reference, frame, allow_reference_metrics=allow_reference_metrics,
                    roi_source=roi_source, measured=measured, psf=psf_items[index],
                )
                for index, frame in enumerate(frames)
            ]
    if diagnostics is not None:
        diagnostics.update(info)
    return result

def metric_score(metrics: Dict[str, float]) -> float:
    """Return a score to maximize for Auto and best-iteration selection."""
    psnr = metrics.get("PSNR", float("nan"))
    if np.isfinite(psnr):
        return float(psnr)
    ssim = metrics.get("SSIM", float("nan"))
    if np.isfinite(ssim):
        return float(100.0 * ssim)
    cost = metrics.get("NO_REFERENCE_COST", float("nan"))
    if np.isfinite(cost):
        return float(-cost)
    ntv = metrics.get("NTV", float("nan"))
    if np.isfinite(ntv):
        return float(-ntv)
    return float("-inf")


def build_intensity_histogram(
    data: np.ndarray,
    bins: int = 4096,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Dict[str, Any]:
    """Build a compact histogram/CDF cache for fast display-level controls.

    The input is flattened only once.  Later conversion between an intensity
    and its approximate percentile is O(1), and quantiles are obtained without
    sorting the image again.
    """
    bins = max(16, int(bins))
    lo, hi = float(value_range[0]), float(value_range[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    arr = np.asarray(data, dtype=np.float32).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        hist = np.zeros(bins, dtype=np.int64)
        hist[0] = 1
        return {
            "hist": hist,
            "cdf": np.cumsum(hist, dtype=np.int64),
            "count": 1,
            "minimum": lo,
            "maximum": hi,
            "range": (lo, hi),
            "bins": bins,
        }
    arr = np.clip(arr, lo, hi)
    hist, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    hist = np.asarray(hist, dtype=np.int64)
    cdf = np.cumsum(hist, dtype=np.int64)
    return {
        "hist": hist,
        "cdf": cdf,
        "count": int(arr.size),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr)),
        "range": (lo, hi),
        "bins": bins,
    }


def combine_intensity_histograms(stats_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine compatible histogram caches without concatenating image arrays."""
    items = [item for item in stats_items if isinstance(item, dict) and item.get("hist") is not None]
    if not items:
        return build_intensity_histogram(np.zeros((1, 1), dtype=np.float32))
    bins = int(items[0]["bins"])
    value_range = tuple(items[0]["range"])
    hist = np.zeros(bins, dtype=np.int64)
    minimum = float("inf")
    maximum = float("-inf")
    count = 0
    for item in items:
        if int(item.get("bins", -1)) != bins or tuple(item.get("range", ())) != value_range:
            raise ValueError("Histogram caches must use the same bins and value range.")
        hist += np.asarray(item["hist"], dtype=np.int64)
        minimum = min(minimum, float(item.get("minimum", value_range[0])))
        maximum = max(maximum, float(item.get("maximum", value_range[1])))
        count += int(item.get("count", 0))
    if count <= 0:
        hist[0] = 1
        count = 1
        minimum, maximum = value_range
    return {
        "hist": hist,
        "cdf": np.cumsum(hist, dtype=np.int64),
        "count": int(count),
        "minimum": float(minimum),
        "maximum": float(maximum),
        "range": value_range,
        "bins": bins,
    }


def histogram_quantile(stats: Dict[str, Any], percentile: float) -> float:
    """Approximate a percentile from a precomputed histogram/CDF."""
    p = float(np.clip(percentile, 0.0, 100.0))
    cdf = np.asarray(stats.get("cdf"), dtype=np.int64)
    count = max(1, int(stats.get("count", int(cdf[-1]) if cdf.size else 1)))
    lo, hi = tuple(stats.get("range", (0.0, 1.0)))
    bins = max(1, int(stats.get("bins", cdf.size if cdf.size else 1)))
    if cdf.size == 0:
        return float(lo)
    target = p * count / 100.0
    index = int(np.searchsorted(cdf, target, side="left"))
    index = min(max(index, 0), bins - 1)
    left_count = int(cdf[index - 1]) if index > 0 else 0
    bin_count = max(1, int(cdf[index]) - left_count)
    fraction = float(np.clip((target - left_count) / bin_count, 0.0, 1.0))
    width = (float(hi) - float(lo)) / bins
    return float(lo + (index + fraction) * width)


def histogram_percentile(stats: Dict[str, Any], value: float) -> float:
    """Approximate the percentile rank of an intensity from a cached CDF."""
    cdf = np.asarray(stats.get("cdf"), dtype=np.int64)
    count = max(1, int(stats.get("count", int(cdf[-1]) if cdf.size else 1)))
    lo, hi = tuple(stats.get("range", (0.0, 1.0)))
    bins = max(1, int(stats.get("bins", cdf.size if cdf.size else 1)))
    if cdf.size == 0 or hi <= lo:
        return 0.0
    position = (float(value) - float(lo)) / (float(hi) - float(lo)) * bins
    if position <= 0.0:
        return 0.0
    if position >= bins:
        return 100.0
    index = int(np.floor(position))
    fraction = position - index
    before = int(cdf[index - 1]) if index > 0 else 0
    in_bin = int(cdf[index]) - before
    rank = before + fraction * in_bin
    return float(np.clip(100.0 * rank / count, 0.0, 100.0))


def optimize_intensity_levels(
    score_function,
    quantile_function,
    current_low: float,
    current_high: float,
) -> Tuple[float, float, float, int]:
    """Optimize display black/white intensities using a small quantile grid.

    Unlike the legacy dense percentile search, this routine evaluates only 39
    candidates: a broad 5x6 grid followed by a local 3x3 refinement.  The
    caller can score downsampled images, keeping the operation responsive.
    """
    cache: Dict[Tuple[int, int], float] = {}

    def canonical(low: float, high: float) -> Tuple[float, float, Tuple[int, int]]:
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))
        key = (int(round(low * 1_000_000)), int(round(high * 1_000_000)))
        return low, high, key

    def evaluate(low: float, high: float) -> Tuple[float, float, float]:
        low, high, key = canonical(low, high)
        if high <= low + 1e-7:
            return low, high, float("-inf")
        if key not in cache:
            try:
                score = float(score_function(low, high))
            except Exception:
                score = float("-inf")
            cache[key] = score if np.isfinite(score) else float("-inf")
        return low, high, cache[key]

    low_ps = (0.0, 0.25, 0.5, 1.0, 2.0)
    high_ps = (95.0, 97.0, 98.5, 99.5, 99.8, 100.0)
    best_low, best_high, best_score = evaluate(current_low, current_high)
    for lp in low_ps:
        low = float(quantile_function(lp))
        for hp in high_ps:
            high = float(quantile_function(hp))
            cand_low, cand_high, score = evaluate(low, high)
            if score > best_score:
                best_low, best_high, best_score = cand_low, cand_high, score

    low_p_center = histogram_percentile_from_callable(quantile_function, best_low)
    high_p_center = histogram_percentile_from_callable(quantile_function, best_high)
    for lp in np.linspace(max(0.0, low_p_center - 0.5), min(5.0, low_p_center + 0.5), 3):
        for hp in np.linspace(max(90.0, high_p_center - 0.5), min(100.0, high_p_center + 0.5), 3):
            cand_low, cand_high, score = evaluate(float(quantile_function(float(lp))), float(quantile_function(float(hp))))
            if score > best_score:
                best_low, best_high, best_score = cand_low, cand_high, score
    return float(best_low), float(best_high), float(best_score), int(len(cache))


def histogram_percentile_from_callable(quantile_function, value: float) -> float:
    """Invert a monotone quantile callable with a short binary search."""
    lo, hi = 0.0, 100.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if float(quantile_function(mid)) < float(value):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def optimize_percentile_range(
    score_function,
    current_low: float = 0.0,
    current_high: float = 97.0,
    low_bounds: Tuple[float, float] = (0.0, 30.0),
    high_bounds: Tuple[float, float] = (60.0, 100.0),
) -> Tuple[float, float, float, int]:
    """Search percentile clipping limits for a score that must be maximized.

    The search is deterministic and uses the same 0.1-percent resolution as the
    GUI sliders.  A broad grid prevents a poor local choice, after which three
    coordinate-refinement passes improve the selected pair without requiring an
    excessive number of expensive image-quality evaluations.
    """
    low_min, low_max = sorted((float(low_bounds[0]), float(low_bounds[1])))
    high_min, high_max = sorted((float(high_bounds[0]), float(high_bounds[1])))
    cache: Dict[Tuple[int, int], float] = {}

    def canonical(low: float, high: float) -> Tuple[float, float, Tuple[int, int]]:
        low = float(np.clip(low, low_min, low_max))
        high = float(np.clip(high, high_min, high_max))
        low_i = int(round(low * 10.0))
        high_i = int(round(high * 10.0))
        low = low_i / 10.0
        high = high_i / 10.0
        return low, high, (low_i, high_i)

    def evaluate(low: float, high: float) -> Tuple[float, float, float]:
        low, high, key = canonical(low, high)
        if high <= low:
            return low, high, float("-inf")
        if key not in cache:
            try:
                score = float(score_function(low, high))
            except Exception:
                score = float("-inf")
            cache[key] = score if np.isfinite(score) else float("-inf")
        return low, high, cache[key]

    low_grid = sorted(set([
        low_min, 0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0,
        float(current_low), low_max,
    ]))
    high_grid = sorted(set([
        high_min, 60.0, 75.0, 85.0, 90.0, 93.0, 95.0, 97.0, 98.0,
        99.0, 99.5, 100.0, float(current_high), high_max,
    ]))
    low_grid = [v for v in low_grid if low_min <= v <= low_max]
    high_grid = [v for v in high_grid if high_min <= v <= high_max]

    best_low, best_high, best_score = evaluate(current_low, current_high)
    for low in low_grid:
        for high in high_grid:
            cand_low, cand_high, score = evaluate(low, high)
            if score > best_score:
                best_low, best_high, best_score = cand_low, cand_high, score

    # Coordinate refinement is considerably cheaper than a dense two-dimensional
    # fine grid, while the broad initial grid still gives global coverage.
    for span in (5.0, 1.0, 0.2):
        low_values = np.linspace(
            max(low_min, best_low - span), min(low_max, best_low + span), 11
        )
        for low in low_values:
            cand_low, cand_high, score = evaluate(float(low), best_high)
            if score > best_score:
                best_low, best_high, best_score = cand_low, cand_high, score

        high_values = np.linspace(
            max(high_min, best_high - span), min(high_max, best_high + span), 11
        )
        for high in high_values:
            cand_low, cand_high, score = evaluate(best_low, float(high))
            if score > best_score:
                best_low, best_high, best_score = cand_low, cand_high, score

    return float(best_low), float(best_high), float(best_score), int(len(cache))


def score_description(score: float, use_reference_metrics: bool) -> str:
    if use_reference_metrics:
        return f"best PSNR {score:.3f} dB" if np.isfinite(score) else "no valid score"
    return f"lowest no-reference cost {-score:.6f}" if np.isfinite(score) else "no valid score"


def describe_psf_kernel(psf: Optional[PSF], label: str = "PSF") -> str:
    """Return compact numerical diagnostics for a PSF."""
    if psf is None:
        return f"{label}: none"
    arr = np.asarray(psf.kernel, dtype=np.float64)
    cy, cx = PSF.center_of_mass(arr)
    return (
        f"{label}: shape={arr.shape[0]}x{arr.shape[1]}, "
        f"sum={float(arr.sum()):.8f}, max={float(arr.max()):.6g}, "
        f"COM=({cy:.2f}, {cx:.2f})"
    )

def compare_psf_kernels(a: Optional[PSF], b: Optional[PSF], label_a: str = "degradation PSF", label_b: str = "reconstruction PSF") -> str:
    """Compare two PSFs after fitting them to a common support."""
    if a is None or b is None:
        return "PSF comparison: unavailable"
    shape = (max(a.kernel.shape[0], b.kernel.shape[0]), max(a.kernel.shape[1], b.kernel.shape[1]))
    ka = a.fitted_to_shape(shape, max_width=max(a.kernel.shape + b.kernel.shape)).kernel
    kb = b.fitted_to_shape(shape, max_width=max(a.kernel.shape + b.kernel.shape)).kernel
    # Put both kernels on the same canvas centered at their centers of mass.
    h = max(ka.shape[0], kb.shape[0])
    w = max(ka.shape[1], kb.shape[1])
    ca = PSF.centered_window(ka, PSF.support_center(ka), h, w)
    cb = PSF.centered_window(kb, PSF.support_center(kb), h, w)
    ca = PSF.normalize_kernel(ca)
    cb = PSF.normalize_kernel(cb)
    diff = ca - cb
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.linalg.norm(ca.ravel()) * np.linalg.norm(cb.ravel()))
    corr = float(np.dot(ca.ravel(), cb.ravel()) / denom) if denom > 1e-15 else float("nan")
    return f"PSF comparison {label_a} vs {label_b}: corr={corr:.6f}, L1={l1:.6g}, RMSE={l2:.6g}"

@dataclass
class DeconvolutionResult:
    image: GrayImage
    history: List[GrayImage] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    info: str = ""

def compute_metrix(
    reference: Optional[GrayImage],
    estimate: Optional[GrayImage],
    allow_reference_metrics: bool = True,
    roi_source: Optional[GrayImage] = None,
    measured: Optional[GrayImage] = None,
    psf: Optional[PSF] = None,
) -> Dict[str, float]:
    """Backward-compatible misspelling alias for older GUI callbacks."""
    return compute_metrics(
        reference, estimate, allow_reference_metrics=allow_reference_metrics,
        roi_source=roi_source, measured=measured, psf=psf,
    )

def normalized_noise_psd_from_image(image: GrayImage, params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return a normalized noise power spectrum for Wiener-type filters.

    The PSD is available when the degraded image was generated by the program.
    In particular, for correlated Gaussian / speckle noise it is computed from
    the actually generated multiplicative-noise disturbance converted to an
    additive-equivalent field: noisy - blurred_clean.  It is normalized to mean
    one so that the Wiener K parameter remains the overall noise-strength scale.
    """
    if not bool(params.get("wiener_use_noise_psd", False)):
        return None
    if image is None or not isinstance(getattr(image, "metadata", None), dict):
        return None
    psd = image.metadata.get("noise_psd")
    if psd is None:
        return None
    arr = np.asarray(psd, dtype=np.float64)
    if arr.shape != image.data.shape:
        arr = _match_array_shape(arr, image.data.shape)
    arr = np.nan_to_num(arr, nan=1.0, posinf=1.0, neginf=1.0)
    arr = np.maximum(arr, 0.0)
    m = float(np.mean(arr))
    if m <= 1e-18:
        return np.ones_like(image.data, dtype=np.float64)
    return arr / m

class DeconvolutionAlgorithm(ABC):
    name: str = "abstract"
    default_params: Dict[str, Any] = {}

    @abstractmethod
    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        raise NotImplementedError

    @staticmethod
    def _report_iteration(params: Dict[str, Any], current: int, total: int) -> None:
        """Report iteration progress to an optional GUI/background callback."""
        callback = params.get("_iteration_callback")
        if callable(callback):
            try:
                callback(int(current), int(total))
            except Exception:
                pass

    @staticmethod
    def _stop_requested(params: Dict[str, Any]) -> bool:
        """Return True when a cooperative stop was requested by the GUI."""
        event = params.get("_stop_event")
        try:
            return bool(event is not None and event.is_set())
        except Exception:
            return False

    @classmethod
    def _iteration_completed(cls, params: Dict[str, Any], current: int, total: int) -> bool:
        """Report a completed iteration and indicate whether to stop before the next one."""
        cls._report_iteration(params, current, total)
        return cls._stop_requested(params)

    @staticmethod
    def _center_crop_array(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Center-crop a 2D array to fit inside target_shape."""
        data = np.asarray(arr, dtype=np.float64)
        th, tw = int(target_shape[0]), int(target_shape[1])
        h, w = data.shape
        ch, cw = min(h, th), min(w, tw)
        y0 = max(0, (h - ch) // 2)
        x0 = max(0, (w - cw) // 2)
        return data[y0:y0 + ch, x0:x0 + cw]

    @staticmethod
    def _pad_psf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Pad a PSF to an FFT shape, cropping it first if it is too large.

        Loaded PSF images may be larger than the calculation image.  In that case
        the physically meaningful central part is used and then sum-normalized.
        """
        shape = (int(shape[0]), int(shape[1]))
        kernel = DeconvolutionAlgorithm._center_crop_array(psf, shape)
        kernel = np.maximum(np.nan_to_num(kernel), 0.0)
        s = float(kernel.sum())
        if s <= 0.0:
            cy, cx = kernel.shape[0] // 2, kernel.shape[1] // 2
            kernel = np.zeros_like(kernel, dtype=np.float64)
            kernel[cy, cx] = 1.0
        else:
            kernel = kernel / s
        return psf_at_fft_origin_numpy(kernel, shape, dtype=np.float64)

    @staticmethod
    def _tv_enabled(params: Dict[str, Any]) -> Tuple[bool, float, int]:
        """Read TV proximal-step parameters shared by iterative algorithms."""
        enabled = bool(params.get("use_tv_preconditioning", False))
        weight = max(0.0, float(params.get("tv_weight", 0.005)))
        iterations = max(1, int(params.get("tv_iterations", 5)))
        return enabled, weight, iterations

    @staticmethod
    def _apply_tv_preconditioner(data: np.ndarray, enabled: bool, weight: float, iterations: int) -> np.ndarray:
        """Apply a Chambolle TV proximal denoising step without changing array shape."""
        if not enabled or weight <= 0.0:
            return data
        arr = np.asarray(data, dtype=np.float64)
        try:
            return denoise_tv_chambolle(
                arr,
                weight=weight,
                max_num_iter=iterations,
                channel_axis=None,
            )
        except TypeError:
            # Compatibility with older scikit-image versions.
            return denoise_tv_chambolle(arr, weight=weight, n_iter_max=iterations)


    @staticmethod
    def _neural_denoiser_mode(params: Dict[str, Any]) -> str:
        """Return neural denoiser mode: off, before, or each_iteration."""
        mode = str(params.get("neural_denoiser_mode", "off")).strip().lower()
        aliases = {
            "off": "off",
            "disabled": "off",
            "before algorithm": "before",
            "before": "before",
            "after each iteration": "each_iteration",
            "each_iteration": "each_iteration",
            "each iteration": "each_iteration",
        }
        return aliases.get(mode, "off")

    @staticmethod
    def _prepare_neural_input(image: GrayImage, params: Dict[str, Any]) -> GrayImage:
        """Optionally denoise the observed input before the deconvolution algorithm starts."""
        if DeconvolutionAlgorithm._neural_denoiser_mode(params) != "before":
            return image
        return GrayImage(neural_denoise_np(image.data, params), name=image.name + "_neural_denoised")

    @staticmethod
    def _apply_neural_iteration_denoiser(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Optionally denoise the current estimate after an algorithm iteration."""
        if DeconvolutionAlgorithm._neural_denoiser_mode(params) != "each_iteration":
            return data
        return neural_denoise_np(data, params)

def torch_device_name(prefer_cuda: bool = True) -> str:
    """Return the selected PyTorch device name."""
    if not TORCH_AVAILABLE:
        return "unavailable"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _torch_image(arr: np.ndarray, device: str, dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
    """Convert a 2D NumPy array to a 1x1xHxW PyTorch tensor.

    Float32 is the default for all Torch computations. Float64 remains
    available as an explicit diagnostic/precision option.
    """
    if dtype is None:
        dtype = torch.float32
    np_dtype = np.float64 if dtype is torch.float64 else np.float32
    return torch.as_tensor(np.asarray(arr, dtype=np_dtype), dtype=dtype, device=device)[None, None, :, :]

def _torch_kernel(kernel: np.ndarray, device: str, flip: bool = True, dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
    """Convert PSF to a convolution/correlation kernel for torch.conv2d."""
    if dtype is None:
        dtype = torch.float32
    np_dtype = np.float64 if dtype is torch.float64 else np.float32
    k = np.asarray(kernel, dtype=np_dtype)
    if flip:
        k = k[::-1, ::-1].copy()
    return torch.as_tensor(k, dtype=dtype, device=device)[None, None, :, :]

def _torch_linear_same_pad(x: "torch.Tensor", kh: int, kw: int) -> "torch.Tensor":
    """Zero-pad ``x`` so a valid conv2d returns exactly the input HxW size.

    For odd kernels the padding is symmetric. For even kernels it is
    intentionally asymmetric and matches ``scipy.signal.fftconvolve(...,
    mode="same")``. Using ``kernel_size // 2`` on both sides would produce an
    output one pixel too large for an even kernel (for example 1281 instead of
    1280).
    """
    top = int(kh // 2)
    bottom = int((kh - 1) // 2)
    left = int(kw // 2)
    right = int((kw - 1) // 2)
    return F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)

def torch_conv_same(x: "torch.Tensor", kernel: np.ndarray, device: str, flip: bool = True) -> "torch.Tensor":
    """Linear same-size 2D convolution for odd or even PSF dimensions."""
    w = _torch_kernel(kernel, device=device, flip=flip, dtype=x.dtype)
    kh, kw = int(w.shape[-2]), int(w.shape[-1])
    return F.conv2d(_torch_linear_same_pad(x, kh, kw), w, padding=0)

def torch_conv_same_tensor(x: "torch.Tensor", kernel: "torch.Tensor", flip: bool = True) -> "torch.Tensor":
    """Differentiable linear same-size convolution for odd or even PSFs.

    ``x`` has shape 1x1xHxW and ``kernel`` has shape KhxKw. The explicit
    asymmetric padding keeps the result exactly HxW and preserves gradients
    with respect to the PSF for blind optimization.
    """
    k = kernel
    if flip:
        k = torch.flip(k, dims=(-2, -1))
    w = k[None, None, :, :]
    kh, kw = int(w.shape[-2]), int(w.shape[-1])
    return F.conv2d(_torch_linear_same_pad(x, kh, kw), w, padding=0)

def torch_project_psf_(p: "torch.Tensor", rotational_symmetry: bool = False) -> None:
    """Project an optimized PSF tensor onto nonnegative, sum-one constraints.

    When rotational_symmetry is enabled, the projection also applies radial
    averaging around the PSF center of mass.  This is intentionally done as a
    projection step rather than a differentiable operation; it keeps the blind
    Adam implementation robust and avoids adding a complicated radial-binning
    autograd graph.
    """
    with torch.no_grad():
        p.clamp_(min=0.0)
        s_val = float(torch.sum(p).detach().cpu().item())
        if s_val <= 1e-18 or not np.isfinite(s_val):
            p.zero_()
            p[p.shape[-2] // 2, p.shape[-1] // 2] = 1.0
        else:
            p.div_(torch.sum(p))
        if rotational_symmetry:
            arr = p.detach().cpu().numpy().astype(np.float64)
            arr = PSF.rotational_project_centered(arr)
            p.copy_(torch.as_tensor(arr, dtype=p.dtype, device=p.device))

def wiener_fft_ifft_numpy(
    data: np.ndarray,
    kernel: np.ndarray,
    k: float,
    noise_psd: Optional[np.ndarray] = None,
    absolute_output: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Wiener deconvolution implemented explicitly with FFT and inverse FFT.

    No dedicated restoration/deconvolution routine is called.  The PSF is
    converted to an OTF on the image grid, the Wiener transfer function is
    formed in the frequency domain, and the result is obtained only through
    ``fft2`` and ``ifft2``.
    """
    np_dtype = np.dtype(dtype)
    y = np.asarray(data, dtype=np_dtype)
    H = psf_to_otf_numpy(np.asarray(kernel, dtype=np_dtype), y.shape, dtype=np_dtype)
    G = fft2(y)
    if noise_psd is None:
        N = np.ones(y.shape, dtype=np_dtype)
    else:
        N = np.asarray(_match_array_shape(np.asarray(noise_psd), y.shape), dtype=np_dtype)
        N = np.maximum(np.nan_to_num(N, nan=1.0, posinf=1.0, neginf=1.0), 0.0)
        mean_noise = float(np.mean(N))
        N = N / mean_noise if mean_noise > 1e-18 else np.ones(y.shape, dtype=np_dtype)
    denominator = np.abs(H) ** 2 + np.asarray(k, dtype=np_dtype) * N
    spectrum = np.conj(H) * G / denominator
    spatial = ifft2(spectrum)
    # The Wiener estimate for real-valued data is the real part of the inverse
    # FFT. ``absolute_output`` is retained only as a deprecated compatibility
    # argument and is intentionally ignored.
    return np.asarray(np.real(spatial), dtype=np_dtype)


def torch_wiener_filter_np(
    data: np.ndarray,
    kernel: np.ndarray,
    k: float,
    device: str = "cpu",
    torch_float64: bool = False,
    noise_psd: Optional[np.ndarray] = None,
    absolute_output: bool = False,
) -> np.ndarray:
    """Return a NumPy Wiener-filtered image, optionally evaluated with Torch.

    The helper follows the same circular-FFT convention as the ordinary Wiener
    implementation. ``noise_psd`` is normalized to mean one so ``k`` remains
    the global regularization scale. ``absolute_output`` is accepted only for
    backward compatibility and is ignored; the real part of IFFT is returned.
    """
    np_dtype = np.float64 if bool(torch_float64) else np.float32
    data_np = np.asarray(data, dtype=np_dtype)
    kernel_np = np.asarray(kernel, dtype=np_dtype)
    k_val = float(k)
    shape = data_np.shape

    noise_np: Optional[np.ndarray]
    if noise_psd is None:
        noise_np = None
    else:
        noise_np = np.asarray(_match_array_shape(np.asarray(noise_psd), shape), dtype=np_dtype)
        noise_np = np.maximum(np.nan_to_num(noise_np, nan=1.0, posinf=1.0, neginf=1.0), 0.0)
        mean_noise = float(np.mean(noise_np))
        noise_np = noise_np / mean_noise if mean_noise > 1e-18 else np.ones(shape, dtype=np_dtype)

    if TORCH_AVAILABLE and device != "unavailable":
        try:
            dtype = torch.float64 if bool(torch_float64) else torch.float32
            y = torch.as_tensor(data_np, dtype=dtype, device=device)
            H = psf_to_otf_torch(kernel_np, shape, device=device, dtype=dtype)
            G = torch.fft.fft2(y)
            if noise_np is None:
                N = torch.ones(shape, dtype=dtype, device=device)
            else:
                N = torch.as_tensor(noise_np, dtype=dtype, device=device)
            F_hat = torch.conj(H) / (torch.abs(H) ** 2 + k_val * N) * G
            spatial = torch.fft.ifft2(F_hat)
            return torch.real(spatial).detach().cpu().numpy().astype(np_dtype)
        except Exception as exc:
            print(f"Torch Wiener initializer failed, falling back to NumPy: {exc}")

    return wiener_fft_ifft_numpy(
        data_np, kernel_np, k_val, noise_psd=noise_np,
        absolute_output=absolute_output, dtype=np_dtype,
    )

def torch_tv_loss(x: "torch.Tensor", isotropic: bool = True, eps: float = 1e-8) -> "torch.Tensor":
    """Differentiable total-variation loss for a 1x1xHxW image tensor."""
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    if isotropic:
        dx_c = dx[..., :-1, :]
        dy_c = dy[..., :, :-1]
        return torch.mean(torch.sqrt(dx_c * dx_c + dy_c * dy_c + eps))
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

def torch_manual_adam_step(x: "torch.Tensor", state: Dict[str, Any], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    """Apply one Adam update without torch.optim.

    Some Python/PyTorch/Triton combinations, especially in interactive IDE
    kernels, may segfault while importing torch.optim because it imports
    torch._dynamo and Triton.  This small implementation avoids that import
    path and keeps the PyTorch algorithm usable on CPU or CUDA.
    """
    if x.grad is None:
        return
    grad = x.grad
    if "m" not in state:
        state["m"] = torch.zeros_like(x)
        state["v"] = torch.zeros_like(x)
        state["t"] = 0
    state["t"] += 1
    t = state["t"]
    m = state["m"]
    v = state["v"]
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    x.addcdiv_(m_hat, torch.sqrt(v_hat).add_(eps), value=-float(lr))

def torch_tv_loss_per_sample(x: "torch.Tensor", isotropic: bool = True, eps: float = 1e-8) -> "torch.Tensor":
    """Return TV loss separately for every batch element.

    Accepts BxHxW or Bx1xHxW tensors and returns a B-vector.  This is used by
    batched Adam Auto, where each candidate may have a different TV weight.
    """
    if x.ndim == 4:
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        reduce_dims = tuple(range(1, dx.ndim))
    else:
        dx = x[:, :, 1:] - x[:, :, :-1]
        dy = x[:, 1:, :] - x[:, :-1, :]
        reduce_dims = (1, 2)
    if isotropic:
        if x.ndim == 4:
            dx_c = dx[..., :-1, :]
            dy_c = dy[..., :, :-1]
            return torch.mean(torch.sqrt(dx_c * dx_c + dy_c * dy_c + eps), dim=tuple(range(1, dx_c.ndim)))
        dx_c = dx[:, :-1, :]
        dy_c = dy[:, :, :-1]
        return torch.mean(torch.sqrt(dx_c * dx_c + dy_c * dy_c + eps), dim=(1, 2))
    return torch.mean(torch.abs(dx), dim=reduce_dims) + torch.mean(torch.abs(dy), dim=reduce_dims)

def torch_manual_adam_step_batched(x: "torch.Tensor", state: Dict[str, Any], lr: "torch.Tensor", active: Optional["torch.Tensor"] = None, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    """Manual Adam update with one learning rate per batch element.

    This avoids torch.optim (which can import Triton/Dynamo and crash in some
    Python 3.13/Spyder environments) while enabling true batched Auto for Adam
    deconvolution candidates.
    """
    if x.grad is None:
        return
    grad = x.grad
    if active is not None:
        while active.ndim < grad.ndim:
            active = active[..., None]
        grad = grad * active.to(dtype=grad.dtype, device=grad.device)
    if "m" not in state:
        state["m"] = torch.zeros_like(x)
        state["v"] = torch.zeros_like(x)
        state["t"] = 0
    state["t"] += 1
    t = state["t"]
    m = state["m"]
    v = state["v"]
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    lr_t = lr.to(dtype=x.dtype, device=x.device)
    while lr_t.ndim < x.ndim:
        lr_t = lr_t[..., None]
    x.addcdiv_(m_hat, torch.sqrt(v_hat).add_(eps), value=-1.0)
    # The line above used unit step; correct it to per-sample LR by undoing and re-applying.
    x.addcdiv_(m_hat, torch.sqrt(v_hat).add_(eps), value=1.0)
    x.add_( -lr_t * (m_hat / (torch.sqrt(v_hat) + eps)) )

def torch_project_psf_batch_(p: "torch.Tensor", rotational_flags: Optional[List[bool]] = None) -> None:
    """Project every PSF in a BxKhxKw tensor to nonnegative sum-one form."""
    with torch.no_grad():
        if p.ndim == 2:
            torch_project_psf_(p, rotational_symmetry=bool(rotational_flags[0]) if rotational_flags else False)
            return
        p.clamp_(min=0.0)
        sums = torch.sum(p, dim=(-2, -1), keepdim=True)
        bad = (~torch.isfinite(sums)) | (sums <= 1e-18)
        p.div_(torch.clamp(sums, min=1e-18))
        if torch.any(bad):
            bad_flat = bad[:, 0, 0]
            p[bad_flat] = 0.0
            p[bad_flat, p.shape[-2] // 2, p.shape[-1] // 2] = 1.0
        if rotational_flags and any(rotational_flags):
            arr = p.detach().cpu().numpy().astype(np.float64)
            for i, flag in enumerate(rotational_flags):
                if flag:
                    arr[i] = PSF.rotational_average(arr[i], center=PSF.center_of_mass(arr[i]))
                    arr[i] = PSF.normalize_kernel(np.maximum(arr[i], 0.0))
            p.copy_(torch.as_tensor(arr, dtype=p.dtype, device=p.device))

class LightweightDnCNNDenoiser(torch.nn.Module if TORCH_AVAILABLE else object):
    """Small DnCNN-style residual denoiser.

    If trained weights are loaded, the network predicts noise and returns x - noise.
    Without external weights, it falls back to a safe Gaussian-like convolutional prior
    implemented as a tiny neural module, so the application remains self-contained.
    """

    def __init__(self, channels: int = 1, features: int = 32, depth: int = 5) -> None:
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        layers = [torch.nn.Conv2d(channels, features, 3, padding=1), torch.nn.ReLU(inplace=True)]
        for _ in range(max(0, depth - 2)):
            layers += [torch.nn.Conv2d(features, features, 3, padding=1), torch.nn.ReLU(inplace=True)]
        layers += [torch.nn.Conv2d(features, channels, 3, padding=1)]
        self.net = torch.nn.Sequential(*layers)
        self.has_trained_weights = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        residual = self.net(x)
        return torch.clamp(x - residual, 0.0, 1.0)

def _denoiser_type(params: Dict[str, Any]) -> str:
    """Return selected denoiser type in a normalized form."""
    kind = str(params.get("denoiser_type", "TV only") or "TV only").strip().lower()
    aliases = {
        "tv only": "tv",
        "tv": "tv",
        "gaussian": "gaussian",
        "bilateral": "bilateral",
        "non-local means": "nlm",
        "non local means": "nlm",
        "nlm": "nlm",
        "wavelet": "wavelet",
        "neural cnn loaded from file": "neural_file",
        "neural cnn": "neural_file",
        "lightweight cnn fallback": "neural_fallback",
        "neural fallback": "neural_fallback",
    }
    return aliases.get(kind, "tv")

def neural_denoise_np(data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply the selected optional denoiser to a 2D NumPy image.

    The timing is controlled elsewhere by ``neural_denoiser_mode``.  This
    function only chooses the operator:
      - TV only: Chambolle TV denoising.
      - Gaussian: local Gaussian smoothing.
      - Bilateral: edge-preserving local smoothing.
      - Non-local Means: patch-based denoising.
      - Wavelet: wavelet shrinkage.
      - Neural CNN loaded from file: DnCNN-style PyTorch model from .pt/.pth.
      - Lightweight CNN fallback: the old self-contained Gaussian-like CNN step.

    ``neural_denoiser_strength`` is intentionally reused as a blend factor for
    all denoisers, so old settings remain compatible.
    """
    arr = np.asarray(data, dtype=np.float64)
    strength = float(np.clip(float(params.get("neural_denoiser_strength", 0.15)), 0.0, 1.0))
    if strength <= 0.0:
        return arr

    kind = _denoiser_type(params)
    denoised = arr

    try:
        if kind == "tv":
            # Use the dedicated TV parameters when available; otherwise map the
            # denoiser strength to a conservative TV weight.
            tv_weight = float(params.get("tv_weight", max(0.001, 0.03 * strength)))
            tv_iter = int(params.get("tv_iterations", 5))
            try:
                denoised = denoise_tv_chambolle(arr, weight=tv_weight, max_num_iter=max(1, tv_iter), channel_axis=None)
            except TypeError:
                denoised = denoise_tv_chambolle(arr, weight=tv_weight, n_iter_max=max(1, tv_iter))
            # TV weight already controls strength; still blend mildly for consistency.
            out = (1.0 - strength) * arr + strength * denoised
            return np.clip(out, 0.0, 1.0).astype(np.float64)

        if kind == "gaussian":
            sigma = max(0.05, 2.0 * strength)
            denoised = gaussian_filter(arr, sigma=sigma)

        elif kind == "bilateral":
            try:
                denoised = denoise_bilateral(arr, sigma_color=max(0.005, 0.15 * strength), sigma_spatial=max(1.0, 5.0 * strength), channel_axis=None)
            except TypeError:
                denoised = denoise_bilateral(arr, sigma_color=max(0.005, 0.15 * strength), sigma_spatial=max(1.0, 5.0 * strength), multichannel=False)

        elif kind == "nlm":
            try:
                sigma_est = float(np.mean(estimate_sigma(arr, channel_axis=None)))
            except TypeError:
                sigma_est = float(np.mean(estimate_sigma(arr, multichannel=False)))
            h = max(0.005, (0.8 + 1.2 * strength) * sigma_est)
            denoised = denoise_nl_means(arr, h=h, fast_mode=True, patch_size=5, patch_distance=6, channel_axis=None)

        elif kind == "wavelet":
            try:
                denoised = denoise_wavelet(arr, sigma=None, method="BayesShrink", mode="soft", rescale_sigma=True, channel_axis=None)
            except TypeError:
                denoised = denoise_wavelet(arr, sigma=None, method="BayesShrink", mode="soft", rescale_sigma=True, multichannel=False)

        elif kind in ("neural_file", "dncnn", "ffdnet", "drunet", "scunet", "model_zoo", "neural_fallback"):
            if not TORCH_AVAILABLE:
                return arr
            prefer_cuda = bool(params.get("prefer_cuda", True))
            device = torch_device_name(prefer_cuda=prefer_cuda)
            if device == "unavailable":
                return arr
            x = _torch_image(np.asarray(arr, dtype=np.float32), device=device)
            weights_path = str(params.get("neural_denoiser_weights", "") or "").strip()
            with torch.no_grad():
                if kind in ("neural_file", "dncnn", "ffdnet", "drunet", "scunet", "model_zoo") and weights_path:
                    try:
                        from deconv.denoisers.model_zoo import load_denoiser_from_file, create_denoiser
                        if kind == "neural_file":
                            # Backwards compatible mode: old lightweight DnCNN architecture.
                            model = LightweightDnCNNDenoiser().to(device)
                            state = torch.load(weights_path, map_location=device)
                            if isinstance(state, dict) and "state_dict" in state:
                                state = state["state_dict"]
                            if isinstance(state, dict):
                                state = {kk.replace("module.", ""): vv for kk, vv in state.items()}
                            model.load_state_dict(state, strict=False)
                        elif kind == "model_zoo":
                            model, _meta = load_denoiser_from_file(weights_path, device=device)
                        else:
                            model = create_denoiser(kind).to(device)
                            state = torch.load(weights_path, map_location=device)
                            if isinstance(state, dict) and "state_dict" in state:
                                state = state["state_dict"]
                            if isinstance(state, dict):
                                state = {kk.replace("module.", ""): vv for kk, vv in state.items()}
                            model.load_state_dict(state, strict=False)
                        model.eval()
                        y = model(x)
                        denoised = y.squeeze().detach().cpu().numpy().astype(np.float64)
                    except Exception as exc:
                        print(f"Model-zoo denoiser failed ({kind}): {exc}")
                        return arr
                else:
                    # Old self-contained fallback: fixed 5x5 Gaussian convolution in torch.
                    kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32, device=device)
                    kernel_2d = torch.outer(kernel_1d, kernel_1d)
                    kernel_2d = kernel_2d / torch.sum(kernel_2d)
                    w = kernel_2d[None, None, :, :]
                    y = F.conv2d(x, w, padding=2)
                    denoised = y.squeeze().detach().cpu().numpy().astype(np.float64)

        else:
            denoised = arr

    except Exception as exc:
        print(f"Denoiser failed ({kind}): {exc}")
        return arr

    out = (1.0 - strength) * arr + strength * np.asarray(denoised, dtype=np.float64)
    return np.clip(out, 0.0, 1.0).astype(np.float64)

@dataclass
class BatchedScores:
    """Scores returned by batched Auto evaluation."""
    scores: List[float]
    infos: List[str] = field(default_factory=list)

def torch_backend_device(prefer_cuda: bool = True) -> str:
    """Return a usable torch device string without importing torch.optim."""
    if not TORCH_AVAILABLE:
        return "unavailable"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _torch_pad_psf_np(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Use the same PSF padding convention as the NumPy FFT algorithms."""
    return DeconvolutionAlgorithm._pad_psf(kernel, shape).astype(np.float32)

def _torch_dtype_from_params(params_list: List[Dict[str, Any]]) -> "torch.dtype":
    """Choose Torch dtype for reference-compatible batched algorithms.

    Torch computations default to float32 for lower memory use and higher GPU
    throughput.  A candidate may explicitly set torch_float64=True when a
    higher-precision comparison with the NumPy reference implementation is
    required.
    """
    use64 = False
    if params_list:
        use64 = bool(params_list[0].get("torch_float64", False))
    return torch.float64 if use64 else torch.float32

def _torch_fft_psf(kernel: np.ndarray, shape: Tuple[int, int], device: str, dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
    if dtype is None:
        dtype = torch.float32
    return psf_to_otf_torch(kernel, shape, device=device, dtype=dtype)

def _torch_batch_values(values: List[Any], default: Any, dtype: str, device: str) -> "torch.Tensor":
    """Create compact parameter tensors without silently promoting to float64.

    Float parameters are float32 by default, matching the image tensors used by
    CUDA algorithms.  Callers running an explicit float64 calculation can cast
    the returned tensor to their working dtype.
    """
    out = [v if v is not None else default for v in values]
    if dtype == "bool":
        out = [1.0 if bool(v) else 0.0 for v in out]
        torch_dtype = torch.float32
    elif dtype == "int":
        out = [int(v) for v in out]
        torch_dtype = torch.int64
    else:
        out = [float(v) for v in out]
        torch_dtype = torch.float32
    return torch.as_tensor(out, dtype=torch_dtype, device=device)

def _torch_batch_wiener(
    y: "torch.Tensor",
    H: "torch.Tensor",
    K: "torch.Tensor",
    absolute_output: Optional["torch.Tensor"] = None,
    noise_psd: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Batched Wiener filter with optional PSD; always return real(IFFT)."""
    G = torch.fft.fft2(y)
    K = K.to(device=y.device, dtype=y.dtype)[:, None, None]
    if noise_psd is None:
        N = torch.ones_like(torch.real(H))
    else:
        N = noise_psd.to(device=y.device, dtype=y.dtype)
        if N.ndim == 2:
            N = N[None, :, :]
        if N.shape[0] == 1 and y.shape[0] > 1:
            N = N.repeat(y.shape[0], 1, 1)
        N = torch.clamp(torch.nan_to_num(N, nan=1.0, posinf=1.0, neginf=1.0), min=0.0)
        means = torch.mean(N, dim=(-2, -1), keepdim=True)
        N = torch.where(means > 1e-18, N / torch.clamp(means, min=1e-18), torch.ones_like(N))
    F_hat = torch.conj(H) / (torch.abs(H) ** 2 + K * N) * G
    spatial = torch.fft.ifft2(F_hat)
    # ``absolute_output`` is a deprecated compatibility argument.  It is
    # intentionally ignored to avoid the nonlinear folding of negative ringing.
    return torch.real(spatial)

def _torch_wiener_option_tensors(
    image: Optional[GrayImage],
    params_list: List[Dict[str, Any]],
    y: "torch.Tensor",
) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    """Build per-candidate auxiliary Wiener controls for a Torch batch."""
    device = str(y.device)
    absolute_output = torch.zeros((max(1, len(params_list)),), dtype=torch.float32, device=device)
    noise_stack: List[np.ndarray] = []
    any_noise = False
    shape = tuple(int(v) for v in y.shape[-2:])
    for p in params_list:
        psd = normalized_noise_psd_from_image(image, p) if image is not None else None
        if psd is None:
            noise_stack.append(np.ones(shape, dtype=np.float32))
        else:
            any_noise = True
            noise_stack.append(np.asarray(psd, dtype=np.float32))
    noise_tensor = None
    if any_noise:
        noise_tensor = torch.as_tensor(np.asarray(noise_stack), dtype=y.dtype, device=y.device)
    return absolute_output, noise_tensor

def _torch_batch_conv(x: "torch.Tensor", H: "torch.Tensor") -> "torch.Tensor":
    return circular_convolve_torch(x, H, adjoint=False)

def _torch_batch_corr(x: "torch.Tensor", H: "torch.Tensor") -> "torch.Tensor":
    return circular_convolve_torch(x, H, adjoint=True)

def _torch_fftconvolve_same_batch(x: "torch.Tensor", kernel: "torch.Tensor") -> "torch.Tensor":
    """Differentiable linear ``same`` convolution for a changing PSF.

    Fixed-PSF iterative algorithms should construct ``TorchLinearSameOperator``
    once and reuse it; this compatibility helper intentionally recomputes the
    PSF spectrum because blind methods optimize the kernel itself.
    """
    if x.ndim < 2:
        raise ValueError("Expected an image tensor with at least two dimensions.")
    return linear_convolve_same_torch(x, kernel)


def _torch_batch_neural_step_np_batch(batch: "torch.Tensor", params_list: List[Dict[str, Any]]) -> "torch.Tensor":
    """Optional neural denoiser step for batched Torch algorithms.

    The denoiser may be a user-loaded PyTorch model or the built-in fallback,
    so this bridge keeps the algorithmic loop batched while applying the
    denoiser candidate-by-candidate only when requested.
    """
    if not any(DeconvolutionAlgorithm._neural_denoiser_mode(p) == "each_iteration" for p in params_list):
        return batch
    device = batch.device
    arr = batch.detach().cpu().numpy()
    out = []
    for i, p in enumerate(params_list):
        img = arr[i]
        if DeconvolutionAlgorithm._neural_denoiser_mode(p) == "each_iteration":
            img = neural_denoise_np(img, p)
        out.append(img)
    return torch.as_tensor(np.asarray(out, dtype=batch.detach().cpu().numpy().dtype), dtype=batch.dtype, device=device)

def _torch_batch_tv_step_np_batch(batch: "torch.Tensor", params_list: List[Dict[str, Any]]) -> "torch.Tensor":
    """Optional TV proximal fallback for batched algorithms.

    TV denoising is still provided by skimage here, so this step runs on CPU and
    is intentionally skipped when disabled. A full Torch TV proximal operator can
    replace this function later without touching GUI or Auto code.
    """
    if not any(bool(p.get("use_tv_preconditioning", False)) for p in params_list):
        return batch
    device = batch.device
    arr = batch.detach().cpu().numpy()
    out = []
    for i, p in enumerate(params_list):
        img = arr[i]
        if bool(p.get("use_tv_preconditioning", False)):
            weight = float(p.get("tv_weight", 0.005))
            iters = int(p.get("tv_iterations", 5))
            try:
                img = denoise_tv_chambolle(img, weight=weight, max_num_iter=iters, channel_axis=None)
            except TypeError:
                img = denoise_tv_chambolle(img, weight=weight, n_iter_max=iters)
        out.append(img)
    return torch.as_tensor(np.asarray(out, dtype=batch.detach().cpu().numpy().dtype), dtype=batch.dtype, device=device)

class TorchBatchedDeconvolutionMixin:
    """Mixin for algorithms that can evaluate many parameter sets at once.

    The calculation tensor layout is BxHxW for deconvolution and Bx1xHxW only
    when an external neural denoiser needs convolutional network input.
    """

    supports_batched_auto = True
    auto_score_every_iteration = True

    def _common_batch_setup(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]]) -> Tuple[str, "torch.Tensor", "torch.Tensor", int, int]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        prefer_cuda = bool(params_list[0].get("prefer_cuda", True)) if params_list else True
        device = torch_backend_device(prefer_cuda=prefer_cuda)
        if device == "unavailable":
            raise RuntimeError("PyTorch is unavailable.")
        torch_dtype = _torch_dtype_from_params(params_list)
        y_np = np.asarray(image.data, dtype=np.float64 if torch_dtype is torch.float64 else np.float32)
        h, w = y_np.shape
        B = max(1, len(params_list))
        y_stack = []
        for p in params_list:
            yy = y_np
            if DeconvolutionAlgorithm._neural_denoiser_mode(p) == "before":
                yy = neural_denoise_np(yy, p).astype(np.float32)
            y_stack.append(yy)
        if not y_stack:
            y_stack = [y_np]
        np_dtype = np.float64 if torch_dtype is torch.float64 else np.float32
        y = torch.as_tensor(np.asarray(y_stack, dtype=np_dtype), dtype=torch_dtype, device=device)
        H_single = _torch_fft_psf(psf.kernel, (h, w), device=device, dtype=torch_dtype)
        H = H_single[None, :, :].repeat(B, 1, 1)
        return device, y, H, h, w

    def _initial_estimate_batch(
        self,
        y: "torch.Tensor",
        H: "torch.Tensor",
        params_list: List[Dict[str, Any]],
        default_k: float = 0.01,
        image: Optional[GrayImage] = None,
    ) -> "torch.Tensor":
        device = y.device
        K = _torch_batch_values([p.get("K", default_k) for p in params_list], default_k, "float", str(device))
        absolute_output, noise_tensor = _torch_wiener_option_tensors(image, params_list, y)
        wien = _torch_batch_wiener(y, H, K, absolute_output=absolute_output, noise_psd=noise_tensor)
        flat = torch.full_like(y, 0.5)
        use_wiener = _torch_batch_values([p.get("begin_with_wiener", False) for p in params_list], False, "bool", str(device))[:, None, None]
        x = use_wiener * wien + (1.0 - use_wiener) * flat
        nonneg = _torch_batch_values([p.get("non_negative", True) for p in params_list], True, "bool", str(device))[:, None, None]
        x = torch.where(nonneg > 0, torch.clamp(x, min=0.0), x)
        return x

    def _finalize_batch(self, x: "torch.Tensor", params_list: List[Dict[str, Any]]) -> "torch.Tensor":
        device = x.device
        nonneg = _torch_batch_values([p.get("non_negative", True) for p in params_list], True, "bool", str(device))[:, None, None]
        x = torch.where(nonneg > 0, torch.clamp(x, min=0.0), x)
        return torch.clamp(x, 0.0, 1.5)

    def _score_batch_tensor(self, reference: GrayImage, x: "torch.Tensor") -> List[float]:
        ref = torch.as_tensor(np.asarray(reference.data, dtype=np.float64 if x.dtype is torch.float64 else np.float32), dtype=x.dtype, device=x.device)[None, :, :]
        ys, xs = original_region_slices(reference, reference.data.shape)
        x_roi = torch.clamp(x[..., ys, xs], 0.0, 1.0)
        ref_roi = ref[..., ys, xs]
        mse = torch.mean((x_roi - ref_roi) ** 2, dim=(-2, -1))
        psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
        return [float(v) for v in psnr.detach().cpu().numpy()]

    def score_batch(self, reference: GrayImage, degraded: GrayImage, psf: PSF, params_list: List[Dict[str, Any]]) -> List[float]:
        """Evaluate many candidate parameter sets and return best PSNR per candidate."""
        results = self.run_batch(degraded, psf, params_list, reference=reference, keep_history=False)
        return results.scores

    def run_batch(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
        raise NotImplementedError

def _adam_batch_device_and_dtype(params_list: List[Dict[str, Any]]) -> Tuple[str, "torch.dtype", np.dtype]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    prefer_cuda = bool(params_list[0].get("prefer_cuda", True)) if params_list else True
    device = torch_backend_device(prefer_cuda=prefer_cuda)
    if device == "unavailable":
        raise RuntimeError("PyTorch is unavailable.")
    use64 = bool(params_list[0].get("torch_float64", False)) if params_list else False
    # Adam uses gradients and may be memory-heavy; float32 is default, float64 is optional.
    return device, (torch.float64 if use64 else torch.float32), (np.float64 if use64 else np.float32)

def _adam_prepare_y_stack(image: GrayImage, params_list: List[Dict[str, Any]], np_dtype: np.dtype) -> np.ndarray:
    y_np = np.asarray(image.data, dtype=np_dtype)
    stack = []
    for p in params_list:
        yy = y_np
        if DeconvolutionAlgorithm._neural_denoiser_mode(p) == "before":
            yy = neural_denoise_np(yy, p).astype(np_dtype)
        stack.append(yy)
    return np.asarray(stack, dtype=np_dtype)

def _score_or_tv_batch(
    reference: Optional[GrayImage],
    x: "torch.Tensor",
    roi_source: Optional[GrayImage] = None,
    measured: Optional["torch.Tensor"] = None,
    psf: Optional["torch.Tensor"] = None,
) -> List[float]:
    source = reference if reference is not None else roi_source
    ys, xs = original_region_slices(source, (int(x.shape[-2]), int(x.shape[-1])))
    x_roi = torch.clamp(x[..., ys, xs], 0.0, 1.0)
    if reference is not None:
        ref = torch.as_tensor(np.asarray(reference.data, dtype=np.float64 if x.dtype is torch.float64 else np.float32), dtype=x.dtype, device=x.device)[None, :, :]
        ref_roi = ref[..., ys, xs]
        mse = torch.mean((x_roi - ref_roi) ** 2, dim=(-2, -1))
        psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
        return [float(v) for v in psnr.detach().cpu().numpy()]

    tv = torch_tv_loss_per_sample(x_roi)
    mean_abs = torch.mean(torch.abs(x_roi), dim=(-2, -1))
    ntv = tv / torch.clamp(mean_abs, min=1e-12)
    intensity_error = torch.zeros_like(ntv)
    residual = torch.zeros_like(ntv)
    whiteness = torch.zeros_like(ntv)
    if measured is not None:
        y = measured.to(dtype=x.dtype, device=x.device)
        y_roi = y[..., ys, xs]
        mean_y = torch.mean(torch.abs(y_roi), dim=(-2, -1))
        mean_x = torch.mean(torch.abs(x_roi), dim=(-2, -1))
        intensity_error = torch.abs(mean_x - mean_y) / torch.clamp(mean_y, min=1e-12)
        if psf is not None:
            if psf.ndim == 2:
                predicted = _torch_fftconvolve_same_batch(x, psf)
            elif psf.ndim == 3:
                predicted = torch.stack([_torch_fftconvolve_same_batch(x[j:j+1], psf[j])[0] for j in range(x.shape[0])], dim=0)
            else:
                predicted = x
        else:
            predicted = x
        pred_roi = predicted[..., ys, xs]
        residual_map = pred_roi - y_roi
        numerator = torch.sqrt(torch.sum(residual_map ** 2, dim=(-2, -1)))
        denominator = torch.sqrt(torch.sum(y_roi ** 2, dim=(-2, -1)))
        residual = numerator / torch.clamp(denominator, min=1e-12)
        centered = residual_map - torch.mean(residual_map, dim=(-2, -1), keepdim=True)
        variance = torch.mean(centered ** 2, dim=(-2, -1))
        corr_terms = []
        for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2)):
            y0a, y1a = max(0, dy), centered.shape[-2] + min(0, dy)
            x0a, x1a = max(0, dx), centered.shape[-1] + min(0, dx)
            y0b, y1b = max(0, -dy), centered.shape[-2] - max(0, dy)
            x0b, x1b = max(0, -dx), centered.shape[-1] - max(0, dx)
            a = centered[..., y0a:y1a, x0a:x1a]
            b = centered[..., y0b:y1b, x0b:x1b]
            rho = torch.mean(a * b, dim=(-2, -1)) / torch.clamp(variance, min=1e-12)
            corr_terms.append(rho ** 2)
        if corr_terms:
            whiteness = torch.mean(torch.stack(corr_terms, dim=0), dim=0)
    cost = residual + 0.005 * ntv + 0.01 * intensity_error + 0.25 * whiteness
    return [float(-v) for v in cost.detach().cpu().numpy()]


# Export shared private helpers as well; algorithm modules use them explicitly.
__all__ = [name for name in globals() if not name.startswith("__")]
