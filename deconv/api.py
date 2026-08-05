"""Public, GUI-independent Python API for image deconvolution.

The functions in this module deliberately avoid importing Qt.  They expose the
same numerical models and registered algorithms that are used by the GUI, so a
script, notebook, batch-processing job, or another application can reuse the
implementation without constructing a main window.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np

from deconv.algorithms.registry import AlgorithmRegistry
from deconv.core.operators import CIRCULAR_FFT, LINEAR_SAME
from deconv.core.runtime import (
    DeconvolutionResult,
    GrayImage,
    PSF,
    calculation_psf_for_image,
    degrade_image,
)

ImageInput = Union[GrayImage, np.ndarray]
PSFInput = Union[PSF, np.ndarray]


def _normalize_unit_interval(array: np.ndarray) -> np.ndarray:
    """Min-max normalize a finite 2D array to [0, 1]."""
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("The image must be a two-dimensional grayscale array.")
    arr = np.nan_to_num(arr)
    minimum = float(arr.min())
    maximum = float(arr.max())
    if maximum > minimum:
        arr = (arr - minimum) / (maximum - minimum)
    elif maximum != 0.0:
        arr = arr / maximum
    return np.clip(arr, 0.0, 1.0)


def as_gray_image(
    image: ImageInput,
    *,
    name: str = "image",
    normalize: bool = False,
) -> GrayImage:
    """Convert a 2D NumPy array to :class:`GrayImage`.

    Parameters
    ----------
    image:
        Existing ``GrayImage`` or a two-dimensional NumPy-compatible array.
    name:
        Descriptive name stored in the data model.
    normalize:
        When ``True``, min-max normalize an arbitrary finite array to ``[0, 1]``.
        When ``False`` (default), values must already lie in ``[0, 1]`` and are
        preserved instead of being contrast-stretched.
    """
    if isinstance(image, GrayImage):
        return image
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("The image must be a two-dimensional grayscale array.")
    if normalize:
        arr = _normalize_unit_interval(arr)
    else:
        if not np.all(np.isfinite(arr)):
            raise ValueError("The image contains non-finite values; use normalize=True or clean the input.")
        tolerance = 1e-12
        if float(arr.min()) < -tolerance or float(arr.max()) > 1.0 + tolerance:
            raise ValueError("Image values must lie in [0, 1] when normalize=False.")
        arr = np.clip(arr, 0.0, 1.0)
    return GrayImage(arr, name=name, metadata={"_preserve_intensity": True})


def as_psf(psf: PSFInput, *, name: str = "psf") -> PSF:
    """Convert a non-negative 2D array to a unit-sum :class:`PSF`."""
    if isinstance(psf, PSF):
        return psf
    arr = np.asarray(psf, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("The PSF must be a two-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("The PSF contains non-finite values.")
    if float(np.max(arr)) <= 0.0:
        raise ValueError("The PSF must contain at least one positive value.")
    return PSF(np.maximum(arr, 0.0), name=name, raw_kernel=np.maximum(arr, 0.0))


def generate_test_image(
    *,
    width: int = 256,
    height: Optional[int] = None,
    padding: int = 0,
    internal_margin_fraction: float = 0.15,
) -> GrayImage:
    """Generate the same synthetic grayscale test image as the GUI."""
    return GrayImage.synthetic(
        width=int(width),
        height=int(width if height is None else height),
        padding=int(padding),
        internal_margin_fraction=float(internal_margin_fraction),
    )


def generate_motion_psf(*, size: int = 21, angle_deg: float = 35.0) -> PSF:
    """Generate a normalized motion-blur PSF, including oblique motion."""
    size = int(size)
    if size <= 0:
        raise ValueError("size must be positive.")
    return PSF.motion(size=size, angle_deg=float(angle_deg))


def disturb_image(
    image: ImageInput,
    psf: PSFInput,
    *,
    noise_sigma: float = 0.01,
    noise_type: str = "Gaussian",
    seed: Optional[int] = None,
    normalize_image: bool = False,
) -> GrayImage:
    """Blur an image with the selected PSF and add a supported disturbance.

    The forward model is the same zero-boundary linear ``same`` convolution as
    in the GUI.  ``seed`` makes synthetic noise reproducible for scripts and
    tests; ``None`` retains nondeterministic behavior.
    """
    image_model = as_gray_image(image, name="input", normalize=normalize_image)
    psf_model = as_psf(psf)
    calculation_psf = calculation_psf_for_image(
        psf_model,
        image_model.data.shape,
        algorithm_convolution_model=LINEAR_SAME,
    )
    if calculation_psf is None:
        raise ValueError("A PSF is required to generate a disturbed image.")
    rng = np.random.default_rng(seed)
    return degrade_image(
        image_model,
        calculation_psf,
        noise_sigma=float(noise_sigma),
        noise_type=str(noise_type),
        rng=rng,
    )


def available_algorithms() -> Tuple[str, ...]:
    """Return canonical names accepted by :func:`run_deconvolution`."""
    return tuple(AlgorithmRegistry().names())


def _resolve_algorithm_name(name: str, registry: AlgorithmRegistry) -> str:
    canonical = registry.names()
    if name in canonical:
        return name
    normalized = str(name).strip().casefold()
    matches = [candidate for candidate in canonical if candidate.casefold() == normalized]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"Unknown algorithm {name!r}. Available algorithms: {', '.join(canonical)}")


def default_parameters(algorithm: str) -> Dict[str, Any]:
    """Return an independent copy of an algorithm's default parameter mapping."""
    registry = AlgorithmRegistry()
    canonical = _resolve_algorithm_name(algorithm, registry)
    return deepcopy(dict(registry.get(canonical).default_params))


def run_deconvolution(
    image: ImageInput,
    psf: Optional[PSFInput],
    *,
    algorithm: str = "Wiener",
    params: Optional[Mapping[str, Any]] = None,
    normalize_image: bool = False,
    **parameter_overrides: Any,
) -> DeconvolutionResult:
    """Run any registered deconvolution algorithm without the GUI.

    ``params`` is merged over the algorithm defaults, followed by keyword
    ``parameter_overrides``.  The return value is the native
    :class:`DeconvolutionResult`; the reconstructed array is available as
    ``result.image.data`` and all recorded iterations as
    ``[frame.data for frame in result.history]``.
    """
    registry = AlgorithmRegistry()
    canonical = _resolve_algorithm_name(algorithm, registry)
    algorithm_object = registry.get(canonical)
    image_model = as_gray_image(image, name="input", normalize=normalize_image)

    calculation_psf: Optional[PSF]
    if psf is None:
        calculation_psf = None
    else:
        psf_model = as_psf(psf)
        convolution_model = CIRCULAR_FFT if canonical in {"Wiener", "Torch batch Wiener"} else LINEAR_SAME
        calculation_psf = calculation_psf_for_image(
            psf_model,
            image_model.data.shape,
            algorithm_convolution_model=convolution_model,
        )

    if calculation_psf is None and not canonical.startswith(("Blind ", "PyTorch Blind ")):
        raise ValueError(f"Algorithm {canonical!r} requires a PSF.")

    merged = deepcopy(dict(algorithm_object.default_params))
    if params is not None:
        merged.update(dict(params))
    merged.update(parameter_overrides)
    return algorithm_object.run(image_model, calculation_psf, **merged)


def wiener_filter(
    image: ImageInput,
    psf: PSFInput,
    *,
    K: float = 0.01,
    non_negative: bool = True,
    use_noise_psd: bool = False,
    normalize_image: bool = False,
) -> DeconvolutionResult:
    """Convenience wrapper for the explicit FFT/IFFT Wiener implementation."""
    return run_deconvolution(
        image,
        psf,
        algorithm="Wiener",
        normalize_image=normalize_image,
        K=float(K),
        non_negative=bool(non_negative),
        wiener_use_noise_psd=bool(use_noise_psd),
    )


def save_grayscale(image: ImageInput, path: Union[str, Path], *, normalize: bool = False) -> Path:
    """Save a ``GrayImage`` or 2D array as an 8-bit grayscale image."""
    model = as_gray_image(image, name=str(path), normalize=normalize)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    return destination


__all__ = [
    "DeconvolutionResult",
    "GrayImage",
    "PSF",
    "as_gray_image",
    "as_psf",
    "available_algorithms",
    "default_parameters",
    "disturb_image",
    "generate_motion_psf",
    "generate_test_image",
    "run_deconvolution",
    "save_grayscale",
    "wiener_filter",
]
