"""Shared convolution and Fourier operators used by deconvolution algorithms.

The module makes boundary conditions explicit:

``linear_same``
    Zero-boundary linear convolution followed by the centered ``same`` crop.
    This is the forward model used to synthesize degraded images and by the
    iterative non-blind algorithms.

``circular_fft``
    Circular convolution on the image grid.  This is the diagonal model used
    by the classical closed-form Wiener filter and by Rosen spectral
    correlation.

Fixed-kernel operators cache the PSF spectrum, which avoids recomputing its FFT
at every iteration.  Torch operators use float32 by default and run on CUDA
when constructed on a CUDA device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.fft import fft2, ifft2, rfft2, irfft2

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch installed
    torch = None
    TORCH_AVAILABLE = False

LINEAR_SAME = "linear_same"
CIRCULAR_FFT = "circular_fft"


def _shape2(shape: Tuple[int, int]) -> Tuple[int, int]:
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {shape}")
    return h, w


def _kernel2_numpy(kernel: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.asarray(kernel, dtype=dtype)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("PSF kernel must be a non-empty 2D array.")
    return np.ascontiguousarray(np.nan_to_num(arr))


def psf_at_fft_origin_numpy(
    kernel: np.ndarray,
    shape: Tuple[int, int],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Embed a centered PSF in ``shape`` and move its center to FFT origin.

    The explicit ``-(size // 2)`` form is intentional.  Writing ``-size // 2``
    shifts an odd kernel one sample too far because floor division is evaluated
    after the unary minus.
    """
    h, w = _shape2(shape)
    k = _kernel2_numpy(kernel, dtype=dtype)
    kh, kw = k.shape
    if kh > h or kw > w:
        raise ValueError(
            f"PSF shape {k.shape} does not fit FFT canvas {(h, w)}. "
            "Fit/crop the PSF before creating a circular operator."
        )
    padded = np.zeros((h, w), dtype=dtype)
    padded[:kh, :kw] = k
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return padded


def psf_to_otf_numpy(
    kernel: np.ndarray,
    shape: Tuple[int, int],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Return the complex OTF for circular convolution on ``shape``."""
    return fft2(psf_at_fft_origin_numpy(kernel, shape, dtype=dtype))


def circular_convolve_numpy(
    image: np.ndarray,
    kernel_or_otf: np.ndarray,
    *,
    kernel_is_otf: bool = False,
    adjoint: bool = False,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Circular convolution/correlation on the image grid."""
    x = np.asarray(image, dtype=dtype)
    if x.ndim != 2:
        raise ValueError("Expected a 2D image.")
    H = np.asarray(kernel_or_otf) if kernel_is_otf else psf_to_otf_numpy(kernel_or_otf, x.shape, dtype=dtype)
    if adjoint:
        H = np.conj(H)
    return np.real(ifft2(fft2(x) * H)).astype(dtype, copy=False)


@dataclass
class NumpyLinearSameOperator:
    """Cached zero-boundary linear convolution and its exact adjoint.

    ``forward(x)`` is equivalent to ``scipy.signal.fftconvolve(x, kernel,
    mode='same')``.  ``adjoint(y)`` is the transpose of that exact cropped
    operator, including the asymmetric convention required by even kernels.
    """

    kernel: np.ndarray
    image_shape: Tuple[int, int]
    dtype: np.dtype = np.float32

    def __post_init__(self) -> None:
        self.image_shape = _shape2(self.image_shape)
        self.kernel = _kernel2_numpy(self.kernel, dtype=self.dtype)
        self.kh, self.kw = self.kernel.shape
        self.h, self.w = self.image_shape
        self.full_shape = (self.h + self.kh - 1, self.w + self.kw - 1)
        self.crop_start = ((self.kh - 1) // 2, (self.kw - 1) // 2)
        self.kernel_spectrum = rfft2(self.kernel, s=self.full_shape)

    def _check(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=self.dtype)
        if arr.shape != self.image_shape:
            raise ValueError(f"Expected image shape {self.image_shape}, got {arr.shape}.")
        return np.ascontiguousarray(np.nan_to_num(arr))

    def forward(self, x: np.ndarray) -> np.ndarray:
        arr = self._check(x)
        full = irfft2(rfft2(arr, s=self.full_shape) * self.kernel_spectrum, s=self.full_shape)
        sy, sx = self.crop_start
        return np.asarray(full[sy:sy + self.h, sx:sx + self.w], dtype=self.dtype)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        arr = self._check(y)
        embedded = np.zeros(self.full_shape, dtype=self.dtype)
        sy, sx = self.crop_start
        embedded[sy:sy + self.h, sx:sx + self.w] = arr
        corr = irfft2(rfft2(embedded, s=self.full_shape) * np.conj(self.kernel_spectrum), s=self.full_shape)
        return np.asarray(corr[:self.h, :self.w], dtype=self.dtype)

    def normal(self, x: np.ndarray) -> np.ndarray:
        return self.adjoint(self.forward(x))


def linear_convolve_same_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """One-shot zero-boundary linear ``same`` convolution."""
    return NumpyLinearSameOperator(kernel, np.asarray(image).shape, dtype=dtype).forward(image)


def linear_correlate_same_numpy(
    image: np.ndarray,
    kernel: np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Apply the exact adjoint of linear ``same`` convolution."""
    return NumpyLinearSameOperator(kernel, np.asarray(image).shape, dtype=dtype).adjoint(image)


def psf_at_fft_origin_torch(
    kernel: Union[np.ndarray, "torch.Tensor"],
    shape: Tuple[int, int],
    *,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """Torch equivalent of :func:`psf_at_fft_origin_numpy`."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    h, w = _shape2(shape)
    if dtype is None:
        dtype = torch.float32
    k = torch.as_tensor(kernel, dtype=dtype, device=device)
    if k.ndim != 2 or k.numel() == 0:
        raise ValueError("PSF kernel must be a non-empty 2D tensor.")
    kh, kw = int(k.shape[-2]), int(k.shape[-1])
    if kh > h or kw > w:
        raise ValueError(f"PSF shape {(kh, kw)} does not fit FFT canvas {(h, w)}.")
    padded = torch.zeros((h, w), dtype=dtype, device=k.device)
    padded[:kh, :kw] = k
    padded = torch.roll(padded, shifts=-(kh // 2), dims=0)
    padded = torch.roll(padded, shifts=-(kw // 2), dims=1)
    return padded


def psf_to_otf_torch(
    kernel: Union[np.ndarray, "torch.Tensor"],
    shape: Tuple[int, int],
    *,
    device: Optional[Union[str, "torch.device"]] = None,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    return torch.fft.fft2(psf_at_fft_origin_torch(kernel, shape, device=device, dtype=dtype))


class TorchLinearSameOperator:
    """Cached Torch/CUDA linear ``same`` operator for a fixed PSF.

    Tensors may have any leading batch/channel dimensions; the final two axes
    must equal ``image_shape``.  Float32 is the default.
    """

    def __init__(
        self,
        kernel: Union[np.ndarray, "torch.Tensor"],
        image_shape: Tuple[int, int],
        *,
        device: Optional[Union[str, "torch.device"]] = None,
        dtype: Optional["torch.dtype"] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        self.h, self.w = _shape2(image_shape)
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.kernel = torch.as_tensor(kernel, dtype=dtype, device=device)
        if self.kernel.ndim != 2 or self.kernel.numel() == 0:
            raise ValueError("PSF kernel must be a non-empty 2D tensor.")
        self.kh, self.kw = int(self.kernel.shape[-2]), int(self.kernel.shape[-1])
        self.full_shape = (self.h + self.kh - 1, self.w + self.kw - 1)
        self.crop_start = ((self.kh - 1) // 2, (self.kw - 1) // 2)
        self.kernel_spectrum = torch.fft.rfft2(self.kernel, s=self.full_shape)

    def _check(self, x: "torch.Tensor") -> "torch.Tensor":
        if tuple(int(v) for v in x.shape[-2:]) != (self.h, self.w):
            raise ValueError(f"Expected trailing image shape {(self.h, self.w)}, got {tuple(x.shape[-2:])}.")
        return x.to(dtype=self.dtype, device=self.kernel.device)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        arr = self._check(x)
        full = torch.fft.irfft2(torch.fft.rfft2(arr, s=self.full_shape) * self.kernel_spectrum, s=self.full_shape)
        sy, sx = self.crop_start
        return full[..., sy:sy + self.h, sx:sx + self.w]

    def adjoint(self, y: "torch.Tensor") -> "torch.Tensor":
        arr = self._check(y)
        embedded = torch.zeros((*arr.shape[:-2], *self.full_shape), dtype=arr.dtype, device=arr.device)
        sy, sx = self.crop_start
        embedded[..., sy:sy + self.h, sx:sx + self.w] = arr
        corr = torch.fft.irfft2(
            torch.fft.rfft2(embedded, s=self.full_shape) * torch.conj(self.kernel_spectrum),
            s=self.full_shape,
        )
        return corr[..., :self.h, :self.w]

    def normal(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.adjoint(self.forward(x))


def linear_convolve_same_torch(
    x: "torch.Tensor",
    kernel: "torch.Tensor",
) -> "torch.Tensor":
    """Differentiable one-shot linear ``same`` convolution.

    Unlike :class:`TorchLinearSameOperator`, this helper recomputes the kernel
    spectrum and therefore supports gradients with respect to a changing PSF.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed.")
    if kernel.ndim != 2:
        raise ValueError("Expected a 2D PSF tensor.")
    h, w = int(x.shape[-2]), int(x.shape[-1])
    kh, kw = int(kernel.shape[-2]), int(kernel.shape[-1])
    full_shape = (h + kh - 1, w + kw - 1)
    X = torch.fft.rfft2(x, s=full_shape)
    K = torch.fft.rfft2(kernel.to(dtype=x.dtype, device=x.device), s=full_shape)
    full = torch.fft.irfft2(X * K, s=full_shape)
    sy, sx = (kh - 1) // 2, (kw - 1) // 2
    return full[..., sy:sy + h, sx:sx + w]


def circular_convolve_torch(
    x: "torch.Tensor",
    otf: "torch.Tensor",
    *,
    adjoint: bool = False,
) -> "torch.Tensor":
    H = torch.conj(otf) if adjoint else otf
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * H))


__all__ = [
    "LINEAR_SAME", "CIRCULAR_FFT", "TORCH_AVAILABLE",
    "psf_at_fft_origin_numpy", "psf_to_otf_numpy", "circular_convolve_numpy",
    "NumpyLinearSameOperator", "linear_convolve_same_numpy", "linear_correlate_same_numpy",
    "psf_at_fft_origin_torch", "psf_to_otf_torch", "TorchLinearSameOperator",
    "linear_convolve_same_torch", "circular_convolve_torch",
]
