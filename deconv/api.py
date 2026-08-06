"""Public, GUI-independent Python API for image deconvolution.

The functions in this module deliberately avoid importing Qt. They expose the
same numerical models, PSF generators, and registered algorithms that are used
by the GUI, so scripts, notebooks, batch-processing jobs, and other
applications can reuse the implementation without constructing a main window.
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

from deconv.optim.auto_api import (
    AutoTuneOptions,
    AutoTuningCancelled,
    AutoTuningResult,
    tune_parameters as _tune_parameters,
)

ImageInput = Union[GrayImage, np.ndarray]
PSFInput = Union[PSF, np.ndarray]

_PSF_GENERATORS = ("gaussian", "motion", "high_frequency", "lens_incoherent")
_TORCH_ALGORITHMS = {
    "Torch batch Wiener",
    "Torch batch Richardson-Lucy",
    "Torch batch Richardson-Lucy-Wiener",
    "Torch batch Richardson-Lucy-Rosen",
    "Torch batch Landweber",
    "PyTorch Adam TV-MAP",
    "PyTorch Blind Adam TV-MAP",
}
_BLIND_ALGORITHMS = {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}


def _normalize_unit_interval(array: np.ndarray) -> np.ndarray:
    """Min-max normalize a finite 2D array to ``[0, 1]``."""
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
    """Convert a 2D array to :class:`GrayImage` without importing the GUI."""
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
    arr = np.maximum(arr, 0.0)
    if float(np.max(arr)) <= 0.0:
        raise ValueError("The PSF must contain at least one positive value.")
    return PSF(arr, name=name, raw_kernel=arr)


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


def generate_gaussian_psf(*, size: int = 21, sigma: float = 3.0) -> PSF:
    """Generate the normalized Gaussian PSF available in the GUI."""
    if int(size) <= 0:
        raise ValueError("size must be positive.")
    if float(sigma) <= 0.0:
        raise ValueError("sigma must be positive.")
    return PSF.gaussian(size=int(size), sigma=float(sigma))


def generate_motion_psf(*, size: int = 21, angle_deg: float = 35.0) -> PSF:
    """Generate a normalized horizontal or oblique motion PSF."""
    if int(size) <= 0:
        raise ValueError("size must be positive.")
    return PSF.motion(size=int(size), angle_deg=float(angle_deg))


def generate_high_frequency_psf(
    *,
    size: int = 21,
    frequency: float = 4.0,
    sigma: float = 4.0,
) -> PSF:
    """Generate the normalized oscillatory high-frequency PSF from the GUI."""
    if int(size) <= 0:
        raise ValueError("size must be positive.")
    if float(frequency) <= 0.0:
        raise ValueError("frequency must be positive.")
    if float(sigma) <= 0.0:
        raise ValueError("sigma must be positive.")
    return PSF.high_frequency(size=int(size), frequency=float(frequency), sigma=float(sigma))


def generate_lens_incoherent_psf(
    *,
    size: int = 65,
    focal_length: float = 0.05,
    distance_before: float = 0.10,
    distance_after: float = 0.10,
    wavelength: float = 550e-9,
    aperture_diameter: float = 0.005,
    diffraction_grid_size: Optional[int] = None,
) -> PSF:
    """Generate the approximate incoherent thin-lens PSF available in the GUI.

    Distances, focal length, wavelength, and aperture diameter are expressed in
    SI units. The returned intensity PSF is non-negative and normalized.
    """
    positive = {
        "size": int(size),
        "focal_length": float(focal_length),
        "distance_before": float(distance_before),
        "distance_after": float(distance_after),
        "wavelength": float(wavelength),
        "aperture_diameter": float(aperture_diameter),
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("All lens-PSF dimensions and distances must be positive.")
    return PSF.lens_incoherent(
        size=int(size),
        focal_length=float(focal_length),
        distance_before=float(distance_before),
        distance_after=float(distance_after),
        wavelength=float(wavelength),
        aperture_diameter=float(aperture_diameter),
        diffraction_grid_size=None if diffraction_grid_size is None else int(diffraction_grid_size),
    )


def available_psf_generators() -> Tuple[str, ...]:
    """Return canonical names accepted by :func:`generate_psf`."""
    return _PSF_GENERATORS


def generate_psf(kind: str, **parameters: Any) -> PSF:
    """Generate any PSF type exposed by the GUI through one dispatcher."""
    normalized = str(kind).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "gaussian": "gaussian",
        "motion": "motion",
        "motion_horizontal": "motion",
        "motion_oblique": "motion",
        "high_frequency": "high_frequency",
        "highfrequency": "high_frequency",
        "lens": "lens_incoherent",
        "lens_incoherent": "lens_incoherent",
        "incoherent_lens": "lens_incoherent",
    }
    canonical = aliases.get(normalized)
    if canonical is None:
        raise KeyError(f"Unknown PSF generator {kind!r}. Available generators: {', '.join(_PSF_GENERATORS)}")
    functions = {
        "gaussian": generate_gaussian_psf,
        "motion": generate_motion_psf,
        "high_frequency": generate_high_frequency_psf,
        "lens_incoherent": generate_lens_incoherent_psf,
    }
    return functions[canonical](**parameters)


def disturb_image(
    image: ImageInput,
    psf: PSFInput,
    *,
    noise_sigma: float = 0.01,
    noise_type: str = "Gaussian",
    seed: Optional[int] = None,
    normalize_image: bool = False,
) -> GrayImage:
    """Apply the GUI forward model and an optional reproducible disturbance."""
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


def available_algorithms(*, include_torch: bool = True) -> Tuple[str, ...]:
    """Return canonical registry names accepted by :func:`run_deconvolution`."""
    names = tuple(AlgorithmRegistry().names())
    if include_torch:
        return names
    return tuple(name for name in names if name not in _TORCH_ALGORITHMS)


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
    psf: Optional[PSFInput] = None,
    *,
    algorithm: str = "Wiener",
    params: Optional[Mapping[str, Any]] = None,
    normalize_image: bool = False,
    **parameter_overrides: Any,
) -> DeconvolutionResult:
    """Run any registered CPU or Torch algorithm without constructing the GUI.

    ``params`` is merged over algorithm defaults, followed by keyword overrides.
    Blind methods may receive ``psf=None``; when an initial known PSF is supplied,
    it is prepared using the same calculation path as in the GUI.
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

    if calculation_psf is None and canonical not in _BLIND_ALGORITHMS:
        raise ValueError(f"Algorithm {canonical!r} requires a PSF.")

    merged = deepcopy(dict(algorithm_object.default_params))
    if params is not None:
        merged.update(dict(params))
    merged.update(parameter_overrides)
    return algorithm_object.run(image_model, calculation_psf, **merged)


def available_auto_strategies() -> Tuple[str, ...]:
    """Return the public Auto search strategies.

    ``quadratic`` mirrors the GUI's fast coordinate search. ``full_batched``
    builds a bounded local Cartesian candidate pool and scores it in batches
    whenever the selected implementation supports batched Auto.
    """
    return ("quadratic", "full_batched")


def auto_tune_parameters(
    image: ImageInput,
    psf: Optional[PSFInput] = None,
    *,
    algorithm: str = "Wiener",
    reference: Optional[ImageInput] = None,
    params: Optional[Mapping[str, Any]] = None,
    auto_options: Optional[AutoTuneOptions | Mapping[str, Any]] = None,
    normalize_image: bool = False,
    normalize_reference: bool = False,
    run_best: bool = False,
    progress_callback: Optional[Any] = None,
    stop_event: Any = None,
    **parameter_overrides: Any,
) -> AutoTuningResult:
    """Select algorithm parameters with the same numerical Auto rules as the GUI.

    Parameters
    ----------
    image:
        Disturbed/measured image to deconvolve.
    psf:
        Known PSF, or an optional initial PSF for blind methods.
    algorithm:
        Any canonical name returned by :func:`available_algorithms`.
    reference:
        Optional independent ground-truth image. When supplied, Auto maximizes
        reconstruction PSNR. Without it, ordinary Wiener uses GCV and the other
        methods use the same no-reference reblur/TV/whiteness criterion as the GUI.
    params and parameter_overrides:
        Initial algorithm settings. Keyword overrides take precedence.
    auto_options:
        :class:`AutoTuneOptions` or a mapping controlling the search strategy and
        which parameter categories may change. Disabled processing stages remain
        frozen exactly as in the GUI.
    run_best:
        When ``True``, execute the requested algorithm once with the accepted
        parameters and store the result in ``AutoTuningResult.deconvolution_result``.
    progress_callback:
        Optional callable receiving human-readable progress messages.
    stop_event:
        Optional object with ``is_set()``. Cancellation is checked between
        candidate evaluations.
    """
    registry = AlgorithmRegistry()
    canonical = _resolve_algorithm_name(algorithm, registry)
    image_model = as_gray_image(image, name="input", normalize=normalize_image)
    reference_model = None if reference is None else as_gray_image(
        reference, name="reference", normalize=normalize_reference
    )
    psf_model = None if psf is None else as_psf(psf)

    merged = deepcopy(dict(registry.get(canonical).default_params))
    if params is not None:
        merged.update(dict(params))
    merged.update(parameter_overrides)

    tuning = _tune_parameters(
        image_model,
        psf_model,
        algorithm=canonical,
        reference=reference_model,
        params=merged,
        auto_options=auto_options,
        registry=registry,
        progress_callback=progress_callback,
        stop_event=stop_event,
    )
    if run_best:
        tuning.deconvolution_result = run_deconvolution(
            image_model,
            psf_model,
            algorithm=canonical,
            params=tuning.best_params,
        )
    return tuning


def auto_deconvolve(
    image: ImageInput,
    psf: Optional[PSFInput] = None,
    **kwargs: Any,
) -> AutoTuningResult:
    """Tune parameters and immediately run the accepted setting.

    This is equivalent to :func:`auto_tune_parameters` with ``run_best=True``.
    The final reconstruction is available as
    ``result.deconvolution_result.image.data``.
    """
    kwargs["run_best"] = True
    return auto_tune_parameters(image, psf, **kwargs)


def _run_named(
    algorithm: str,
    image: ImageInput,
    psf: Optional[PSFInput],
    *,
    normalize_image: bool = False,
    **params: Any,
) -> DeconvolutionResult:
    return run_deconvolution(
        image,
        psf,
        algorithm=algorithm,
        normalize_image=normalize_image,
        **params,
    )


# CPU convenience wrappers -------------------------------------------------

def wiener_filter(image: ImageInput, psf: PSFInput, *, K: float = 0.01,
                  non_negative: bool = True, use_noise_psd: bool = False,
                  normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run the explicit FFT/IFFT Wiener implementation."""
    return _run_named("Wiener", image, psf, normalize_image=normalize_image,
                      K=float(K), non_negative=bool(non_negative),
                      wiener_use_noise_psd=bool(use_noise_psd), **options)


def richardson_lucy_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 20,
                           epsilon: float = 1e-8, non_negative: bool = True,
                           begin_with_wiener: bool = False, K: float = 0.01,
                           normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run classical Richardson-Lucy deconvolution."""
    return _run_named("Richardson-Lucy", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), epsilon=float(epsilon), K=float(K),
                      non_negative=bool(non_negative), begin_with_wiener=bool(begin_with_wiener),
                      **options)


def richardson_lucy_wiener_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 20,
                                  epsilon: float = 1e-8, K: float = 0.01,
                                  non_negative: bool = True, begin_with_wiener: bool = False,
                                  normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run the Richardson-Lucy-Wiener hybrid."""
    return _run_named("Richardson-Lucy-Wiener", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), epsilon=float(epsilon), K=float(K),
                      non_negative=bool(non_negative), begin_with_wiener=bool(begin_with_wiener),
                      **options)


def richardson_lucy_rosen_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 20,
                                 epsilon: float = 1e-8, rosen_L: float = 0.5,
                                 rosen_M: float = 0.5, rosen_relax_to_one: bool = False,
                                 rosen_relax_factor: float = 0.98, non_negative: bool = True,
                                 begin_with_wiener: bool = False, K: float = 0.01,
                                 normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run the Richardson-Lucy-Rosen implementation used by the GUI."""
    return _run_named("Richardson-Lucy-Rosen", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), epsilon=float(epsilon), rosen_L=float(rosen_L),
                      rosen_M=float(rosen_M), rosen_relax_to_one=bool(rosen_relax_to_one),
                      rosen_relax_factor=float(rosen_relax_factor), non_negative=bool(non_negative),
                      begin_with_wiener=bool(begin_with_wiener), K=float(K), **options)


def blind_richardson_lucy_filter(image: ImageInput, psf: Optional[PSFInput] = None, *,
                                 iterations: int = 20, epsilon: float = 1e-8,
                                 psf_height: int = 0, psf_width: int = 0,
                                 psf_sigma: float = 3.0, use_known_psf_init: bool = True,
                                 rotational_symmetry: bool = False, non_negative: bool = True,
                                 normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run blind Richardson-Lucy, optionally initialized by a supplied PSF."""
    return _run_named("Blind Richardson-Lucy", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), epsilon=float(epsilon),
                      blind_psf_height=int(psf_height), blind_psf_width=int(psf_width),
                      psf_sigma=float(psf_sigma), blind_use_known_psf_init=bool(use_known_psf_init),
                      blind_psf_rotational_symmetry=bool(rotational_symmetry),
                      non_negative=bool(non_negative), **options)


def landweber_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 50,
                     step: float = 0.8, non_negative: bool = True,
                     begin_with_wiener: bool = False, K: float = 0.01,
                     normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run Landweber iteration."""
    return _run_named("Landweber", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), step=float(step), K=float(K),
                      non_negative=bool(non_negative), begin_with_wiener=bool(begin_with_wiener),
                      **options)


def landweber_wiener_preconditioned_filter(image: ImageInput, psf: PSFInput, *,
                                           iterations: int = 50, step: float = 0.8,
                                           K: float = 0.01, non_negative: bool = True,
                                           begin_with_wiener: bool = False,
                                           normalize_image: bool = False,
                                           **options: Any) -> DeconvolutionResult:
    """Run Wiener-preconditioned Landweber iteration."""
    return _run_named("Landweber Wiener-preconditioned", image, psf,
                      normalize_image=normalize_image, iterations=int(iterations),
                      step=float(step), K=float(K), non_negative=bool(non_negative),
                      begin_with_wiener=bool(begin_with_wiener), **options)


def block_kaczmarz_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 30,
                          relaxation: float = 0.15, block_size: int = 32,
                          blocks_per_iteration: int = 16, full_sweep: bool = True,
                          overlap: bool = True, randomized: bool = False,
                          shift_grid: bool = True, smooth_window: bool = True,
                          stabilized_sweep: bool = True, update_damping: float = 0.5,
                          max_update_fraction: float = 0.25, non_negative: bool = True,
                          begin_with_wiener: bool = False, K: float = 0.01,
                          normalize_image: bool = False, **options: Any) -> DeconvolutionResult:
    """Run the block Kaczmarz/ART implementation."""
    return _run_named("Block Kaczmarz", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), kaczmarz_relaxation=float(relaxation),
                      kaczmarz_block_size=int(block_size),
                      kaczmarz_blocks_per_iteration=int(blocks_per_iteration),
                      kaczmarz_full_sweep=bool(full_sweep), kaczmarz_overlap=bool(overlap),
                      kaczmarz_randomized=bool(randomized), kaczmarz_shift_grid=bool(shift_grid),
                      kaczmarz_window=bool(smooth_window),
                      kaczmarz_stabilized_sweep=bool(stabilized_sweep),
                      kaczmarz_update_damping=float(update_damping),
                      kaczmarz_max_update_fraction=float(max_update_fraction),
                      non_negative=bool(non_negative), begin_with_wiener=bool(begin_with_wiener),
                      K=float(K), **options)


# Torch convenience wrappers ----------------------------------------------

def torch_wiener_filter(image: ImageInput, psf: PSFInput, *, K: float = 0.01,
                        non_negative: bool = True, prefer_cuda: bool = True,
                        float64: bool = False, normalize_image: bool = False,
                        **options: Any) -> DeconvolutionResult:
    return _run_named("Torch batch Wiener", image, psf, normalize_image=normalize_image,
                      K=float(K), non_negative=bool(non_negative), prefer_cuda=bool(prefer_cuda),
                      torch_float64=bool(float64), **options)


def torch_richardson_lucy_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 20,
                                 epsilon: float = 1e-8, prefer_cuda: bool = True,
                                 float64: bool = False, normalize_image: bool = False,
                                 **options: Any) -> DeconvolutionResult:
    return _run_named("Torch batch Richardson-Lucy", image, psf,
                      normalize_image=normalize_image, iterations=int(iterations),
                      epsilon=float(epsilon), prefer_cuda=bool(prefer_cuda),
                      torch_float64=bool(float64), **options)


def torch_richardson_lucy_wiener_filter(image: ImageInput, psf: PSFInput, *,
                                        iterations: int = 20, epsilon: float = 1e-8,
                                        K: float = 0.01, prefer_cuda: bool = True,
                                        float64: bool = False, normalize_image: bool = False,
                                        **options: Any) -> DeconvolutionResult:
    return _run_named("Torch batch Richardson-Lucy-Wiener", image, psf,
                      normalize_image=normalize_image, iterations=int(iterations),
                      epsilon=float(epsilon), K=float(K), prefer_cuda=bool(prefer_cuda),
                      torch_float64=bool(float64), **options)


def torch_richardson_lucy_rosen_filter(image: ImageInput, psf: PSFInput, *,
                                       iterations: int = 20, epsilon: float = 1e-8,
                                       rosen_L: float = 0.5, rosen_M: float = 0.5,
                                       prefer_cuda: bool = True, float64: bool = False,
                                       normalize_image: bool = False,
                                       **options: Any) -> DeconvolutionResult:
    return _run_named("Torch batch Richardson-Lucy-Rosen", image, psf,
                      normalize_image=normalize_image, iterations=int(iterations),
                      epsilon=float(epsilon), rosen_L=float(rosen_L), rosen_M=float(rosen_M),
                      prefer_cuda=bool(prefer_cuda), torch_float64=bool(float64), **options)


def torch_landweber_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 50,
                           step: float = 0.8, prefer_cuda: bool = True,
                           float64: bool = False, normalize_image: bool = False,
                           **options: Any) -> DeconvolutionResult:
    return _run_named("Torch batch Landweber", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), step=float(step), prefer_cuda=bool(prefer_cuda),
                      torch_float64=bool(float64), **options)


def torch_adam_tv_map_filter(image: ImageInput, psf: PSFInput, *, iterations: int = 100,
                             learning_rate: float = 0.05, tv_weight: float = 0.002,
                             prefer_cuda: bool = True, float64: bool = False,
                             normalize_image: bool = False,
                             **options: Any) -> DeconvolutionResult:
    return _run_named("PyTorch Adam TV-MAP", image, psf, normalize_image=normalize_image,
                      iterations=int(iterations), torch_lr=float(learning_rate),
                      tv_weight=float(tv_weight), prefer_cuda=bool(prefer_cuda),
                      torch_float64=bool(float64), **options)


def torch_blind_adam_tv_map_filter(image: ImageInput, psf: Optional[PSFInput] = None, *,
                                   iterations: int = 150, learning_rate: float = 0.03,
                                   psf_learning_rate: float = 0.01,
                                   tv_weight: float = 0.002,
                                   psf_tv_weight: float = 0.0005,
                                   psf_height: int = 0, psf_width: int = 0,
                                   psf_sigma: float = 4.0,
                                   use_known_psf_init: bool = True,
                                   rotational_symmetry: bool = False,
                                   prefer_cuda: bool = True, float64: bool = False,
                                   normalize_image: bool = False,
                                   **options: Any) -> DeconvolutionResult:
    return _run_named("PyTorch Blind Adam TV-MAP", image, psf,
                      normalize_image=normalize_image, iterations=int(iterations),
                      torch_lr=float(learning_rate), blind_psf_lr=float(psf_learning_rate),
                      tv_weight=float(tv_weight), blind_psf_tv_weight=float(psf_tv_weight),
                      blind_psf_height=int(psf_height), blind_psf_width=int(psf_width),
                      psf_sigma=float(psf_sigma), blind_use_known_psf_init=bool(use_known_psf_init),
                      blind_psf_rotational_symmetry=bool(rotational_symmetry),
                      prefer_cuda=bool(prefer_cuda), torch_float64=bool(float64), **options)


def save_grayscale(image: ImageInput, path: Union[str, Path], *, normalize: bool = False) -> Path:
    """Save a :class:`GrayImage` or 2D array as an 8-bit grayscale image."""
    model = as_gray_image(image, name=str(path), normalize=normalize)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    return destination


__all__ = [
    "DeconvolutionResult", "GrayImage", "PSF", "ImageInput", "PSFInput",
    "as_gray_image", "as_psf", "available_algorithms", "available_psf_generators",
    "available_auto_strategies", "AutoTuneOptions", "AutoTuningCancelled", "AutoTuningResult",
    "auto_tune_parameters", "auto_deconvolve",
    "default_parameters", "disturb_image", "generate_psf", "generate_test_image",
    "generate_gaussian_psf", "generate_motion_psf", "generate_high_frequency_psf",
    "generate_lens_incoherent_psf", "run_deconvolution", "save_grayscale",
    "wiener_filter", "richardson_lucy_filter", "richardson_lucy_wiener_filter",
    "richardson_lucy_rosen_filter", "blind_richardson_lucy_filter",
    "landweber_filter", "landweber_wiener_preconditioned_filter", "block_kaczmarz_filter",
    "torch_wiener_filter", "torch_richardson_lucy_filter",
    "torch_richardson_lucy_wiener_filter", "torch_richardson_lucy_rosen_filter",
    "torch_landweber_filter", "torch_adam_tv_map_filter", "torch_blind_adam_tv_map_filter",
]
