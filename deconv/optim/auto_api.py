"""GUI-independent automatic parameter tuning.

This module exposes the numerical core of the GUI's ``Auto`` feature without
importing Qt.  It deliberately uses the same algorithm registry, candidate
ranges, parameter-activation rules and reconstruction criteria as the GUI.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import itertools
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from deconv.algorithms.registry import AlgorithmRegistry
from deconv.core.operators import CIRCULAR_FFT, LINEAR_SAME
from deconv.core.runtime import (
    DeconvolutionResult,
    GrayImage,
    PSF,
    TORCH_AVAILABLE,
    auto_tunable_parameter_names,
    calculation_psf_for_image,
    compute_metrics,
    metric_score,
    normalized_noise_psd_from_image,
    score_description,
    wiener_gcv_cost,
)

ProgressCallback = Callable[[str], None]

_TORCH_BATCH_PAIRS = {
    "Wiener": "Torch batch Wiener",
    "Richardson-Lucy": "Torch batch Richardson-Lucy",
    "Richardson-Lucy-Wiener": "Torch batch Richardson-Lucy-Wiener",
    "Richardson-Lucy-Rosen": "Torch batch Richardson-Lucy-Rosen",
    "Landweber": "Torch batch Landweber",
}
_TORCH_BATCH_REVERSE = {value: key for key, value in _TORCH_BATCH_PAIRS.items()}
_BLIND_ALGORITHMS = {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}
_DIRECT_WIENER_K_ALGORITHMS = {
    "Wiener",
    "Torch batch Wiener",
    "Richardson-Lucy-Wiener",
    "Torch batch Richardson-Lucy-Wiener",
    "Landweber Wiener-preconditioned",
}


@dataclass(frozen=True)
class AutoTuneOptions:
    """Controls for :func:`deconv.api.auto_tune_parameters`.

    The defaults mirror the conservative GUI Auto settings.  Feature-enabling
    controls are frozen at the beginning of a run, so Auto cannot silently turn
    on a disabled Wiener initializer, denoiser or optional TV stage.
    """

    strategy: str = "quadratic"
    max_candidates: int = 256
    batch_size: int = 32
    passes: int = 2
    tune_numeric: bool = True
    tune_boolean: bool = False
    tune_categorical: bool = False
    tune_denoiser: bool = False
    tune_denoiser_strength: bool = True
    tune_tv: bool = True
    tune_wiener_init: bool = False
    use_torch_equivalent: bool = True
    validate_on_requested_algorithm: bool = True

    @classmethod
    def from_value(cls, value: Optional["AutoTuneOptions | Mapping[str, Any]"]) -> "AutoTuneOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            known = {field_name for field_name in cls.__dataclass_fields__}
            unknown = sorted(set(value) - known)
            if unknown:
                raise TypeError(f"Unknown Auto option(s): {', '.join(unknown)}")
            return cls(**dict(value))
        raise TypeError("auto_options must be AutoTuneOptions, a mapping, or None.")


@dataclass
class AutoTuningResult:
    """Result returned by the public Auto API."""

    requested_algorithm: str
    tuned_algorithm: str
    initial_params: Dict[str, Any]
    best_params: Dict[str, Any]
    initial_score: float
    best_score: float
    score_label: str
    evaluations: int
    elapsed_seconds: float
    status: str
    deconvolution_result: Optional[DeconvolutionResult] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return bool(np.isfinite(self.best_score) and (
            not np.isfinite(self.initial_score) or self.best_score > self.initial_score + 1e-12
        ))


class AutoTuningCancelled(RuntimeError):
    """Raised when a supplied stop event requests cancellation."""


class _HeadlessAutoTuner:
    def __init__(
        self,
        image: GrayImage,
        psf: Optional[PSF],
        *,
        reference: Optional[GrayImage],
        requested_algorithm: str,
        initial_params: Mapping[str, Any],
        options: AutoTuneOptions,
        registry: Optional[AlgorithmRegistry] = None,
        progress_callback: Optional[ProgressCallback] = None,
        stop_event: Any = None,
    ) -> None:
        self.image = image
        self.psf = psf
        self.reference = reference
        self.requested_algorithm = str(requested_algorithm)
        self.options = options
        self.registry = registry or AlgorithmRegistry()
        self.progress_callback = progress_callback
        self.stop_event = stop_event
        self.evaluations = 0
        self.history: List[Dict[str, Any]] = []

        if reference is not None and tuple(reference.data.shape) != tuple(image.data.shape):
            raise ValueError("reference and image must have the same shape.")
        if self.requested_algorithm not in self.registry.names():
            raise KeyError(f"Unknown algorithm {requested_algorithm!r}.")
        if psf is None and self.requested_algorithm not in _BLIND_ALGORITHMS:
            raise ValueError(f"Algorithm {self.requested_algorithm!r} requires a PSF.")

        self.tuned_algorithm = self._choose_tuned_algorithm(self.requested_algorithm)
        self.public_initial_params = self._prepare_blind_shape(dict(initial_params))
        self._requested_param_keys = set(self.public_initial_params)
        tuned_defaults = deepcopy(dict(self.registry.get(self.tuned_algorithm).default_params))
        requested_defaults = deepcopy(dict(self.registry.get(self.requested_algorithm).default_params))
        tuned_defaults.update(requested_defaults)
        tuned_defaults.update(self.public_initial_params)
        self.initial_params = self._prepare_blind_shape(tuned_defaults)
        self._allowed_parameter_names = auto_tunable_parameter_names(
            self.tuned_algorithm,
            self._active_parameter_names(self.tuned_algorithm),
            self.initial_params,
        )

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(str(message))

    def _check_cancel(self) -> None:
        event = self.stop_event
        if event is not None and callable(getattr(event, "is_set", None)) and event.is_set():
            raise AutoTuningCancelled("Auto parameter tuning was cancelled.")

    def _choose_tuned_algorithm(self, requested: str) -> str:
        if not self.options.use_torch_equivalent or not TORCH_AVAILABLE:
            return requested
        candidate = _TORCH_BATCH_PAIRS.get(requested)
        if candidate and candidate in self.registry.names():
            return candidate
        return requested

    def _prepare_blind_shape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        if self.requested_algorithm not in _BLIND_ALGORITHMS and self.tuned_algorithm not in _BLIND_ALGORITHMS:
            return params
        if self.psf is not None:
            default_h, default_w = self.psf.kernel.shape
        else:
            linked = max(3, int(round(0.45 * min(self.image.data.shape))))
            default_h = default_w = linked
        if int(params.get("blind_psf_height", 0) or 0) <= 0:
            params["blind_psf_height"] = int(default_h)
        if int(params.get("blind_psf_width", 0) or 0) <= 0:
            params["blind_psf_width"] = int(default_w)
        return params

    @staticmethod
    def _strip_internal(params: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in dict(params).items() if not str(key).startswith("__auto_")}

    def _public_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        clean = self._strip_internal(params)
        if self.tuned_algorithm == self.requested_algorithm:
            return clean
        return {key: value for key, value in clean.items() if key in self._requested_param_keys}

    def _calculation_psf(self, algorithm: str, params: Mapping[str, Any]) -> Optional[PSF]:
        is_blind = algorithm in _BLIND_ALGORITHMS
        if is_blind and not bool(params.get("blind_use_known_psf_init", True)):
            return None
        if self.psf is None:
            return None
        model = CIRCULAR_FFT if algorithm in {"Wiener", "Torch batch Wiener"} else LINEAR_SAME
        prepared = calculation_psf_for_image(
            self.psf,
            self.image.data.shape,
            algorithm_convolution_model=model,
        )
        if prepared is None:
            return None
        meta = dict(prepared.metadata or {})
        meta.update({
            "algorithm_convolution_model": model,
            "convolution_model": model,
            "wiener_kernel_source": "public_api_psf",
        })
        return PSF(prepared.kernel.copy(), name=prepared.name, raw_kernel=prepared.raw_kernel, metadata=meta)

    @staticmethod
    def _estimated_psf_from_frame(frame: GrayImage, fallback: Optional[PSF]) -> Optional[PSF]:
        metadata = getattr(frame, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("estimated_psf") is not None:
            try:
                return PSF(np.asarray(metadata["estimated_psf"], dtype=np.float64), name="estimated_psf_for_auto")
            except Exception:
                pass
        return fallback

    def _score_description(self, algorithm: str, score: float) -> str:
        if algorithm in {"Wiener", "Torch batch Wiener"} and self.reference is None:
            return f"lowest Wiener GCV {-score:.6g}" if np.isfinite(score) else "no valid Wiener GCV"
        return score_description(score, self.reference is not None)

    def _score_result(self, result: DeconvolutionResult, run_psf: Optional[PSF]) -> float:
        frames = result.history or [result.image]
        scores: List[float] = []
        for frame in frames:
            quality_psf = self._estimated_psf_from_frame(frame, run_psf)
            metrics = compute_metrics(
                self.reference,
                frame,
                allow_reference_metrics=self.reference is not None,
                roi_source=self.image,
                measured=self.image,
                psf=quality_psf,
            )
            scores.append(metric_score(metrics))
        return float(max(scores)) if scores else float("-inf")

    def _plain_wiener_gcv_score(self, algorithm: str, params: Mapping[str, Any], run_psf: Optional[PSF]) -> Optional[float]:
        if algorithm not in {"Wiener", "Torch batch Wiener"} or self.reference is not None or run_psf is None:
            return None
        try:
            noise_psd = normalized_noise_psd_from_image(self.image, dict(params))
            value = wiener_gcv_cost(self.image.data, run_psf, float(params.get("K", 0.01)), noise_psd=noise_psd)
        except Exception:
            return float("-inf")
        return float(-value) if np.isfinite(value) else float("-inf")

    def _score_one(self, algorithm: str, params: Mapping[str, Any]) -> float:
        self._check_cancel()
        clean = self._strip_internal(params)
        run_psf = self._calculation_psf(algorithm, clean)
        if run_psf is None and algorithm not in _BLIND_ALGORITHMS:
            return float("-inf")
        gcv_score = self._plain_wiener_gcv_score(algorithm, clean, run_psf)
        self.evaluations += 1
        if gcv_score is not None:
            score = float(gcv_score)
        else:
            algorithm_object = self.registry.get(algorithm)
            safe_params = dict(clean)
            if algorithm == "PyTorch Adam TV-MAP":
                safe_params["iterations"] = min(int(safe_params.get("iterations", 100)), 25)
            try:
                result = algorithm_object.run(self.image, run_psf, **safe_params)
                score = self._score_result(result, run_psf)
            except Exception:
                score = float("-inf")
        self.history.append({"algorithm": algorithm, "params": clean, "score": score})
        return score

    def _score_batch(self, algorithm: str, candidates: Sequence[Mapping[str, Any]]) -> List[float]:
        self._check_cancel()
        if not candidates:
            return []
        clean_candidates = [self._strip_internal(candidate) for candidate in candidates]
        algorithm_object = self.registry.get(algorithm)
        if not bool(getattr(algorithm_object, "supports_batched_auto", False)):
            return [self._score_one(algorithm, candidate) for candidate in clean_candidates]
        run_psf = self._calculation_psf(algorithm, clean_candidates[0])
        if run_psf is None and algorithm not in _BLIND_ALGORITHMS:
            return [float("-inf")] * len(clean_candidates)
        if algorithm in {"Wiener", "Torch batch Wiener"} and self.reference is None:
            values = [
                float(self._plain_wiener_gcv_score(algorithm, candidate, run_psf) or float("-inf"))
                for candidate in clean_candidates
            ]
            self.evaluations += len(clean_candidates)
            for candidate, score in zip(clean_candidates, values):
                self.history.append({"algorithm": algorithm, "params": dict(candidate), "score": float(score)})
            return values

        batch_size = max(1, int(self.options.batch_size))
        scores: List[float] = []
        for start in range(0, len(clean_candidates), batch_size):
            self._check_cancel()
            group = clean_candidates[start:start + batch_size]
            used_scalar_fallback = False
            try:
                if self.reference is not None:
                    group_scores = list(algorithm_object.score_batch(self.reference, self.image, run_psf, group))
                else:
                    batched = algorithm_object.run_batch(self.image, run_psf, group, reference=None, keep_history=False)
                    group_scores = []
                    for item in batched.infos:
                        if isinstance(item, dict):
                            array = np.asarray(item.get("image"), dtype=np.float64)
                            estimated = item.get("estimated_psf")
                        else:
                            array = np.asarray(item, dtype=np.float64)
                            estimated = None
                        frame = GrayImage(array, name="auto_candidate")
                        quality_psf = PSF(np.asarray(estimated), name="auto_estimated_psf") if estimated is not None else run_psf
                        metrics = compute_metrics(
                            None,
                            frame,
                            allow_reference_metrics=False,
                            roi_source=self.image,
                            measured=self.image,
                            psf=quality_psf,
                        )
                        group_scores.append(metric_score(metrics))
            except Exception:
                used_scalar_fallback = True
                group_scores = [self._score_one(algorithm, candidate) for candidate in group]
            if not used_scalar_fallback:
                self.evaluations += len(group)
                for candidate, score in zip(group, group_scores):
                    self.history.append({"algorithm": algorithm, "params": dict(candidate), "score": float(score)})
            scores.extend(float(value) for value in group_scores)
        return scores

    def _active_parameter_names(self, algorithm: str) -> List[str]:
        denoiser = ["neural_denoiser_mode", "denoiser_type", "neural_denoiser_strength", "neural_denoiser_weights"]
        tv = ["use_tv_preconditioning", "tv_weight", "tv_iterations"]
        wiener_controls = ["wiener_use_noise_psd"]
        common_iter = ["iterations", "epsilon", "begin_with_wiener", "K", "non_negative"] + tv + denoiser
        mapping = {
            "Wiener": ["K", "non_negative"] + wiener_controls + tv + denoiser,
            "Richardson-Lucy": common_iter + wiener_controls,
            "Richardson-Lucy-Wiener": ["iterations", "epsilon", "K", "begin_with_wiener", "non_negative"] + wiener_controls + tv + denoiser,
            "Blind Richardson-Lucy": ["iterations", "epsilon", "psf_sigma", "blind_psf_rotational_symmetry", "blind_use_known_psf_init", "begin_with_wiener", "K", "non_negative"] + wiener_controls + tv + denoiser,
            "Landweber": ["iterations", "step", "begin_with_wiener", "K", "non_negative"] + wiener_controls + tv + denoiser,
            "Landweber Wiener-preconditioned": ["iterations", "step", "K", "begin_with_wiener", "non_negative"] + wiener_controls + tv + denoiser,
            "Block Kaczmarz": ["iterations", "kaczmarz_relaxation", "kaczmarz_block_size", "kaczmarz_blocks_per_iteration", "kaczmarz_full_sweep", "kaczmarz_overlap", "kaczmarz_randomized", "kaczmarz_shift_grid", "kaczmarz_window", "kaczmarz_stabilized_sweep", "kaczmarz_update_damping", "kaczmarz_max_update_fraction", "begin_with_wiener", "K", "non_negative"] + wiener_controls + tv + denoiser,
            "Richardson-Lucy-Rosen": ["iterations", "epsilon", "K", "rosen_L", "rosen_M", "rosen_relax_to_one", "rosen_relax_factor", "begin_with_wiener", "non_negative"] + wiener_controls + tv + denoiser,
            "PyTorch Adam TV-MAP": ["iterations", "torch_lr", "tv_weight", "begin_with_wiener", "K", "non_negative", "prefer_cuda", "torch_float64", "torch_record_every"] + wiener_controls + denoiser,
            "PyTorch Blind Adam TV-MAP": ["iterations", "torch_lr", "blind_psf_lr", "tv_weight", "blind_psf_tv_weight", "psf_sigma", "blind_psf_rotational_symmetry", "blind_use_known_psf_init", "begin_with_wiener", "K", "non_negative", "prefer_cuda", "torch_float64", "torch_record_every"] + wiener_controls + denoiser,
            "Torch batch Wiener": ["K", "non_negative", "prefer_cuda", "torch_float64"] + wiener_controls,
            "Torch batch Richardson-Lucy": ["iterations", "epsilon", "begin_with_wiener", "K", "non_negative", "prefer_cuda", "torch_float64"] + wiener_controls + denoiser,
            "Torch batch Richardson-Lucy-Wiener": ["iterations", "epsilon", "K", "begin_with_wiener", "non_negative", "prefer_cuda", "torch_float64"] + wiener_controls + denoiser,
            "Torch batch Richardson-Lucy-Rosen": ["iterations", "epsilon", "K", "rosen_L", "rosen_M", "rosen_relax_to_one", "rosen_relax_factor", "begin_with_wiener", "non_negative", "prefer_cuda", "torch_float64"] + wiener_controls + tv + denoiser,
            "Torch batch Landweber": ["iterations", "step", "K", "begin_with_wiener", "non_negative", "prefer_cuda", "torch_float64"] + wiener_controls + denoiser,
        }
        return list(mapping.get(algorithm, common_iter))

    def _candidate_values(self, name: str, value: Any) -> List[Any]:
        if name not in set(self._allowed_parameter_names):
            return []
        technical = {"prefer_cuda", "torch_float64", "torch_record_every", "neural_denoiser_weights"}
        if name in technical:
            return []
        tv_names = {"use_tv_preconditioning", "tv_weight", "tv_iterations"}
        denoiser_names = {"neural_denoiser_mode", "denoiser_type", "neural_denoiser_strength"}
        if name in tv_names and not self.options.tune_tv:
            return []
        if name == "begin_with_wiener" and not self.options.tune_wiener_init:
            return []
        if name in denoiser_names:
            if name == "neural_denoiser_strength":
                if not (self.options.tune_denoiser or self.options.tune_denoiser_strength):
                    return []
            elif not (self.options.tune_denoiser or self.options.tune_categorical):
                return []
        if isinstance(value, bool):
            return [value, not value] if self.options.tune_boolean else [value]
        if isinstance(value, str):
            if not (self.options.tune_categorical or (name in denoiser_names and self.options.tune_denoiser)):
                return [value]
            if name == "neural_denoiser_mode":
                return list(dict.fromkeys([value, "Off", "Before algorithm", "After each iteration"]))
            if name == "denoiser_type":
                return list(dict.fromkeys([value, "TV only", "Gaussian", "Bilateral", "Non-local Means", "Wavelet"]))
            return [value]
        if (
            not self.options.tune_numeric
            and name not in tv_names
            and name not in denoiser_names
            and name != "K"
        ):
            return []
        if name == "K" and not (self.options.tune_numeric or self.options.tune_wiener_init):
            return []
        if value is None:
            return []
        integer_names = {"iterations", "kaczmarz_block_size", "kaczmarz_blocks_per_iteration", "tv_iterations", "torch_record_every"}
        if name in integer_names:
            return sorted({max(1, int(round(float(value) * factor))) for factor in (0.5, 0.75, 1.0, 1.25, 1.5)})
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return [value]
        if numeric == 0.0:
            return [0.0]
        return [numeric * factor for factor in (0.5, 0.75, 1.0, 1.25, 1.5)]

    @staticmethod
    def _coerce_value(name: str, value: float) -> Any:
        bounds: Dict[str, Tuple[float, float]] = {
            "K": (1e-12, 1e4),
            "iterations": (1.0, 100000.0),
            "kaczmarz_block_size": (2.0, 4096.0),
            "kaczmarz_blocks_per_iteration": (1.0, 100000.0),
            "tv_iterations": (1.0, 10000.0),
            "torch_record_every": (1.0, 100000.0),
            "epsilon": (1e-15, 1.0),
            "torch_lr": (1e-12, 10.0),
            "blind_psf_lr": (1e-12, 10.0),
            "tv_weight": (0.0, 100.0),
            "blind_psf_tv_weight": (0.0, 100.0),
            "step": (1e-12, 100.0),
        }
        lo, hi = bounds.get(name, (-1e12, 1e12))
        clipped = float(np.clip(float(value), lo, hi))
        if name in {"iterations", "kaczmarz_block_size", "kaczmarz_blocks_per_iteration", "tv_iterations", "torch_record_every"}:
            return max(1, int(round(clipped)))
        return clipped

    @staticmethod
    def _quadratic_vertex(name: str, xs: Sequence[float], ys: Sequence[float]) -> Optional[Any]:
        finite = sorted({float(x): float(y) for x, y in zip(xs, ys) if np.isfinite(y)}.items())
        if len(finite) < 3:
            return None
        x_array = np.asarray([item[0] for item in finite], dtype=np.float64)
        y_array = np.asarray([item[1] for item in finite], dtype=np.float64)
        center = float(np.mean(x_array))
        scale = float(np.max(np.abs(x_array - center)))
        if scale <= 1e-18:
            return None
        z = (x_array - center) / scale
        try:
            a, b, _ = np.polyfit(z, y_array, 2)
        except Exception:
            return None
        if not np.isfinite(a) or not np.isfinite(b) or a >= -1e-14:
            return None
        x_star = center + scale * float(-b / (2.0 * a))
        if not np.isfinite(x_star) or x_star <= float(np.min(x_array)) or x_star >= float(np.max(x_array)):
            return None
        candidate = _HeadlessAutoTuner._coerce_value(name, x_star)
        if any(abs(float(candidate) - x) <= 1e-14 * max(1.0, abs(x)) for x in x_array):
            return None
        return candidate

    @staticmethod
    def _unique_positive(values: Iterable[float], lo: float = 1e-12, hi: float = 1e4) -> List[float]:
        output: List[float] = []
        for raw in values:
            value = float(np.clip(float(raw), lo, hi))
            if not np.isfinite(value) or value <= 0:
                continue
            if not any(abs(math.log10(value) - math.log10(old)) < 1e-10 for old in output):
                output.append(value)
        return sorted(output)

    def _optimize_k(self, algorithm: str, params: Dict[str, Any], initial_score: float) -> Tuple[Dict[str, Any], float]:
        if "K" not in self._allowed_parameter_names or not (
            self.options.tune_numeric or self.options.tune_wiener_init
        ):
            return dict(params), float(initial_score)
        if algorithm not in _DIRECT_WIENER_K_ALGORITHMS and not bool(params.get("begin_with_wiener", False)):
            return dict(params), float(initial_score)
        lo, hi = 1e-12, 1e4
        current = float(params.get("K", 0.01))
        coarse = [current]
        for exponent in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1):
            base = 10.0 ** exponent
            coarse.extend((base, math.sqrt(10.0) * base))
        best_params, best_score = dict(params), float(initial_score)

        def evaluate(values: Sequence[float]) -> None:
            nonlocal best_params, best_score
            trials = []
            for value in self._unique_positive(values, lo, hi):
                if abs(math.log10(value) - math.log10(max(float(best_params.get("K", current)), 1e-300))) < 1e-12:
                    continue
                trial = dict(best_params)
                trial["K"] = value
                trials.append(trial)
            for trial, score in zip(trials, self._score_batch(algorithm, trials)):
                if np.isfinite(score) and score > best_score:
                    best_params, best_score = trial, float(score)

        evaluate(coarse)
        for half_span, count in ((0.50, 11), (0.12, 9)):
            center = float(best_params.get("K", current))
            factors = 10.0 ** np.linspace(-half_span, half_span, count)
            evaluate([center * float(factor) for factor in factors] + [center])
        return best_params, best_score

    def _quadratic_coordinate(self, algorithm: str, initial: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        best_params = dict(initial)
        best_score = self._score_one(algorithm, best_params)
        best_params, best_score = self._optimize_k(algorithm, best_params, best_score)
        active_names = [name for name in self._active_parameter_names(algorithm) if name != "K"]
        for _ in range(max(1, int(self.options.passes))):
            changed = False
            for name in active_names:
                self._check_cancel()
                values = list(dict.fromkeys(self._candidate_values(name, best_params.get(name))))
                if len(values) <= 1:
                    continue
                current = best_params.get(name)
                numeric = isinstance(current, (int, float, np.integer, np.floating)) and not isinstance(current, bool)
                if numeric:
                    ordered = sorted(values, key=float)
                    below = [value for value in ordered if float(value) < float(current)]
                    above = [value for value in ordered if float(value) > float(current)]
                    samples = ([below[-1]] if below else []) + [current] + ([above[0]] if above else [])
                    samples = list(dict.fromkeys(samples))
                else:
                    samples = values
                trials: List[Dict[str, Any]] = []
                scores: List[float] = []
                pending: List[Dict[str, Any]] = []
                pending_indices: List[int] = []
                for value in samples:
                    trial = dict(best_params)
                    trial[name] = value
                    trials.append(trial)
                    if value == current:
                        scores.append(best_score)
                    else:
                        pending_indices.append(len(scores))
                        scores.append(float("nan"))
                        pending.append(trial)
                if pending:
                    pending_scores = self._score_batch(algorithm, pending)
                    for position, score in zip(pending_indices, pending_scores):
                        scores[position] = float(score)
                if numeric and len(trials) >= 3:
                    vertex = self._quadratic_vertex(name, [float(trial[name]) for trial in trials], scores)
                    if vertex is not None:
                        vertex_trial = dict(best_params)
                        vertex_trial[name] = vertex
                        vertex_score = self._score_batch(algorithm, [vertex_trial])[0]
                        trials.append(vertex_trial)
                        scores.append(float(vertex_score))
                finite = np.asarray(scores, dtype=np.float64)
                if finite.size and np.isfinite(finite).any():
                    index = int(np.nanargmax(finite))
                    if float(scores[index]) > best_score:
                        best_params = trials[index]
                        best_score = float(scores[index])
                        changed = True
            if not changed:
                break
        return best_params, best_score

    def _candidate_pool(self, initial: Dict[str, Any]) -> List[Dict[str, Any]]:
        names: List[str] = []
        value_lists: List[List[Any]] = []
        for name in self._active_parameter_names(self.tuned_algorithm):
            if name == "K":
                continue
            values = list(dict.fromkeys(self._candidate_values(name, initial.get(name))))
            if not values:
                continue
            names.append(name)
            value_lists.append(values)
        if not value_lists:
            return [dict(initial)]
        candidates: List[Dict[str, Any]] = []
        seen = set()
        limit = max(1, min(int(self.options.max_candidates), 4096))
        for combo in itertools.product(*value_lists):
            if len(candidates) >= limit:
                break
            self._check_cancel()
            trial = dict(initial)
            for name, value in zip(names, combo):
                trial[name] = value
            signature = tuple((name, trial.get(name)) for name in names)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(trial)
        return candidates or [dict(initial)]

    def _full_batched(self, algorithm: str, initial: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        baseline = self._score_one(algorithm, initial)
        k_params, k_score = self._optimize_k(algorithm, initial, baseline)
        candidates = self._candidate_pool(k_params)
        scores = self._score_batch(algorithm, candidates)
        if not scores or not np.isfinite(np.asarray(scores, dtype=np.float64)).any():
            return k_params, k_score
        index = int(np.nanargmax(np.asarray(scores, dtype=np.float64)))
        if float(scores[index]) > k_score:
            return candidates[index], float(scores[index])
        return k_params, k_score

    def tune(self) -> AutoTuningResult:
        start = time.perf_counter()
        self._emit(f"Auto: scoring baseline for {self.requested_algorithm} ...")
        validation_algorithm = self.requested_algorithm
        initial_clean = self._public_params(self.public_initial_params)
        initial_score = self._score_one(validation_algorithm, initial_clean)

        self._emit(f"Auto: tuning {self.tuned_algorithm} ...")
        strategy = self.options.strategy.strip().lower().replace("-", "_").replace(" ", "_")
        if strategy in {"quadratic", "quadratic_coordinate", "coordinate", "fast"}:
            best_params, proxy_score = self._quadratic_coordinate(self.tuned_algorithm, dict(self.initial_params))
            strategy_name = "quadratic coordinate"
        elif strategy in {"full", "full_batched", "batched", "cartesian"}:
            best_params, proxy_score = self._full_batched(self.tuned_algorithm, dict(self.initial_params))
            strategy_name = "full batched"
        else:
            raise ValueError("strategy must be 'quadratic' or 'full_batched'.")

        best_params = self._strip_internal(best_params)
        if self.options.validate_on_requested_algorithm and self.tuned_algorithm != validation_algorithm:
            self._emit(f"Auto: validating the best candidate on {validation_algorithm} ...")
            best_score = self._score_one(validation_algorithm, best_params)
        else:
            best_score = float(proxy_score)

        tolerance = 1e-9
        if np.isfinite(initial_score) and (not np.isfinite(best_score) or best_score < initial_score - tolerance):
            status = (
                f"Auto kept the initial parameters for {validation_algorithm}; "
                f"the best candidate ({self._score_description(validation_algorithm, best_score)}) "
                f"was worse than the baseline ({self._score_description(validation_algorithm, initial_score)})."
            )
            accepted_params = initial_clean
            accepted_score = initial_score
        else:
            status = (
                f"Auto ({strategy_name}) selected parameters for {validation_algorithm}: "
                f"{self._score_description(validation_algorithm, best_score)}; "
                f"evaluated {self.evaluations} candidate runs."
            )
            accepted_params = self._public_params(best_params)
            accepted_score = best_score

        elapsed = time.perf_counter() - start
        self._emit(status)
        return AutoTuningResult(
            requested_algorithm=self.requested_algorithm,
            tuned_algorithm=self.tuned_algorithm,
            initial_params=initial_clean,
            best_params=accepted_params,
            initial_score=float(initial_score),
            best_score=float(accepted_score),
            score_label=self._score_description(validation_algorithm, accepted_score),
            evaluations=int(self.evaluations),
            elapsed_seconds=float(elapsed),
            status=status,
            history=list(self.history),
        )


def tune_parameters(
    image: GrayImage,
    psf: Optional[PSF],
    *,
    algorithm: str,
    reference: Optional[GrayImage],
    params: Mapping[str, Any],
    auto_options: Optional[AutoTuneOptions | Mapping[str, Any]] = None,
    registry: Optional[AlgorithmRegistry] = None,
    progress_callback: Optional[ProgressCallback] = None,
    stop_event: Any = None,
) -> AutoTuningResult:
    """Internal typed entry point used by :mod:`deconv.api`."""
    tuner = _HeadlessAutoTuner(
        image,
        psf,
        reference=reference,
        requested_algorithm=algorithm,
        initial_params=params,
        options=AutoTuneOptions.from_value(auto_options),
        registry=registry,
        progress_callback=progress_callback,
        stop_event=stop_event,
    )
    return tuner.tune()


__all__ = [
    "AutoTuneOptions",
    "AutoTuningCancelled",
    "AutoTuningResult",
    "tune_parameters",
]
