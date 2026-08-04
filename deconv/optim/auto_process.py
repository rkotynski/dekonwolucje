"""Isolated numerical executor for Auto/Auto All.

The Qt worker orchestrates parameter searches, while potentially long numerical
candidate evaluations run in this separate process.  Cancellation is cooperative
at algorithm iteration boundaries.  If a candidate does not return within the
configured grace period, the helper process can be terminated without killing a
Qt/Python worker thread or leaving the GUI's numerical lock held.
"""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from deconv.algorithms import AlgorithmRegistry
from deconv.core.runtime import (
    GrayImage,
    PSF,
    compute_metrics,
    metric_score,
    normalized_noise_psd_from_image,
    wiener_gcv_cost,
)


class AutoCancelledError(RuntimeError):
    """Raised in the orchestrating worker when Auto cancellation completes."""


class AutoProcessError(RuntimeError):
    """Raised when the isolated numerical process cannot complete a request."""


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message or "cufft" in message)


def _torch_cleanup() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass


def _quality_psf_from_frame(frame: GrayImage, fallback: Optional[PSF]) -> Optional[PSF]:
    metadata = getattr(frame, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("estimated_psf") is not None:
        try:
            return PSF(np.asarray(metadata["estimated_psf"], dtype=np.float64), name="estimated_psf_for_quality")
        except Exception:
            pass
    return fallback


def _metrics(
    reference: Optional[GrayImage],
    degraded: GrayImage,
    frame: GrayImage,
    fallback_psf: Optional[PSF],
    allow_reference: bool,
) -> Dict[str, float]:
    return compute_metrics(
        reference if allow_reference else None,
        frame,
        allow_reference_metrics=bool(allow_reference),
        roi_source=degraded,
        measured=degraded,
        psf=_quality_psf_from_frame(frame, fallback_psf),
    )


def _score_one(
    registry: AlgorithmRegistry,
    reference: Optional[GrayImage],
    degraded: GrayImage,
    allow_reference: bool,
    stop_event: Any,
    alg_name: str,
    params: Dict[str, Any],
    run_psf: Optional[PSF],
) -> float:
    if stop_event.is_set():
        return float("-inf")
    alg = registry.get(str(alg_name))
    is_blind = str(alg_name) in {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}
    if run_psf is None and not is_blind:
        return float("-inf")

    if str(alg_name) in {"Wiener", "Torch batch Wiener"} and not allow_reference:
        try:
            noise_psd = normalized_noise_psd_from_image(degraded, params)
            value = wiener_gcv_cost(degraded.data, run_psf, float(params.get("K", 0.01)), noise_psd=noise_psd)
            return float(-value) if np.isfinite(value) else float("-inf")
        except Exception:
            return float("-inf")

    safe_params = dict(params)
    safe_params["_stop_event"] = stop_event
    if str(alg_name) == "PyTorch Adam TV-MAP":
        safe_params["iterations"] = min(int(safe_params.get("iterations", 100)), 25)
    if is_blind and not bool(safe_params.get("blind_use_known_psf_init", True)):
        effective_psf = None
    else:
        effective_psf = run_psf

    try:
        result = alg.run(degraded, effective_psf, **safe_params)
    except Exception as exc:
        if _is_cuda_oom(exc) and bool(safe_params.get("prefer_cuda", True)) and not stop_event.is_set():
            _torch_cleanup()
            retry = dict(safe_params)
            retry["prefer_cuda"] = False
            try:
                result = alg.run(degraded, effective_psf, **retry)
            except Exception:
                return float("-inf")
        else:
            return float("-inf")

    frames = result.history or [result.image]
    scores = [metric_score(_metrics(reference, degraded, frame, run_psf, allow_reference)) for frame in frames]
    return float(max(scores)) if scores else float("-inf")


def _score_batch(
    registry: AlgorithmRegistry,
    reference: Optional[GrayImage],
    degraded: GrayImage,
    allow_reference: bool,
    stop_event: Any,
    alg_name: str,
    candidates: List[Dict[str, Any]],
    run_psf: Optional[PSF],
) -> List[float]:
    if not candidates:
        return []
    if stop_event.is_set():
        return [float("-inf")] * len(candidates)
    alg = registry.get(str(alg_name))
    is_blind = str(alg_name) in {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}
    if run_psf is None and not is_blind:
        return [float("-inf")] * len(candidates)

    prepared: List[Dict[str, Any]] = []
    for params in candidates:
        trial = dict(params)
        trial["_stop_event"] = stop_event
        prepared.append(trial)

    if str(alg_name) in {"Wiener", "Torch batch Wiener"} and not allow_reference:
        return [
            _score_one(registry, reference, degraded, allow_reference, stop_event, alg_name, p, run_psf)
            for p in prepared
        ]

    if not bool(getattr(alg, "supports_batched_auto", False)):
        return [
            _score_one(registry, reference, degraded, allow_reference, stop_event, alg_name, p, run_psf)
            for p in prepared
        ]

    effective_psf = run_psf
    if is_blind and not bool(prepared[0].get("blind_use_known_psf_init", True)):
        effective_psf = None
    try:
        if allow_reference:
            return [float(v) for v in alg.score_batch(reference, degraded, effective_psf, prepared)]
        batched = alg.run_batch(degraded, effective_psf, prepared, reference=None, keep_history=False)
        values: List[float] = []
        for item in batched.infos:
            if isinstance(item, dict):
                arr = np.asarray(item.get("image"), dtype=np.float64)
                estimated = item.get("estimated_psf")
                quality_psf = PSF(np.asarray(estimated, dtype=np.float64), name="candidate_estimated_psf") if estimated is not None else run_psf
            else:
                arr = np.asarray(item, dtype=np.float64)
                quality_psf = run_psf
            frame = GrayImage(arr, name="candidate")
            values.append(float(metric_score(_metrics(None, degraded, frame, quality_psf, False))))
        return values
    except Exception as exc:
        if _is_cuda_oom(exc) and not stop_event.is_set():
            _torch_cleanup()
            if len(prepared) > 1:
                middle = max(1, len(prepared) // 2)
                return (
                    _score_batch(registry, reference, degraded, allow_reference, stop_event, alg_name, prepared[:middle], run_psf)
                    + _score_batch(registry, reference, degraded, allow_reference, stop_event, alg_name, prepared[middle:], run_psf)
                )
            cpu_params = dict(prepared[0])
            cpu_params["prefer_cuda"] = False
            return [_score_one(registry, reference, degraded, allow_reference, stop_event, alg_name, cpu_params, run_psf)]
        return [
            _score_one(registry, reference, degraded, allow_reference, stop_event, alg_name, p, run_psf)
            for p in prepared
        ]


def auto_numerical_process_main(connection: Any, payload: Dict[str, Any], stop_event: Any) -> None:
    """Entry point for the isolated numerical process."""
    registry = AlgorithmRegistry()
    reference = payload.get("reference")
    degraded = payload.get("degraded")
    allow_reference = bool(payload.get("allow_reference", False))
    default_psf = payload.get("psf")
    try:
        # Let the parent know that the spawned process has finished importing
        # the numerical stack and is ready to receive a command.  Starting the
        # cancellation grace period only after this handshake prevents process
        # start-up latency from being mistaken for an unresponsive iteration.
        connection.send({"ok": True, "event": "ready"})
        while True:
            command = connection.recv()
            operation = str(command.get("op", ""))
            if operation == "shutdown":
                connection.send({"ok": True, "value": None})
                break
            if operation == "sleep":  # Small deterministic hook used by cancellation tests.
                deadline = time.monotonic() + max(0.0, float(command.get("seconds", 0.0)))
                while time.monotonic() < deadline:
                    if stop_event.is_set():
                        # Deliberately remain alive when requested by the test so the
                        # parent hard-stop path can also be verified.
                        if not bool(command.get("ignore_cancel", False)):
                            break
                    time.sleep(0.02)
                connection.send({"ok": True, "value": "slept"})
                continue
            if not isinstance(degraded, GrayImage):
                connection.send({"ok": False, "error": "Auto numerical process has no degraded/measured image."})
                continue
            if operation == "score_one":
                value = _score_one(
                    registry, reference, degraded, allow_reference, stop_event,
                    str(command["algorithm"]), dict(command["params"]), command.get("psf", default_psf),
                )
                connection.send({"ok": True, "value": float(value)})
                continue
            if operation == "score_batch":
                value = _score_batch(
                    registry, reference, degraded, allow_reference, stop_event,
                    str(command["algorithm"]), [dict(p) for p in command["params_list"]], command.get("psf", default_psf),
                )
                connection.send({"ok": True, "value": [float(v) for v in value]})
                continue
            connection.send({"ok": False, "error": f"Unknown Auto numerical operation: {operation}"})
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        _torch_cleanup()
        try:
            connection.close()
        except Exception:
            pass


@dataclass
class AutoNumericalProcessClient:
    """Persistent client used by one Auto/Auto All worker."""

    payload: Dict[str, Any]
    cancel_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        event = context.Event()
        process = context.Process(
            target=auto_numerical_process_main,
            args=(child, dict(self.payload), event),
            name="DeconvolutionAutoNumericalProcess",
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._stop_event = event
        self._process = process
        self._cancel_requested_at: Optional[float] = None
        self._forced = False
        self._closed = False

        # Wait for a deterministic child-ready handshake.  Without it, a user
        # cancellation (or a fast test timer) could begin while the spawned
        # interpreter was still importing modules, consuming the entire grace
        # period before the first numerical command had even started.
        try:
            if not self._connection.poll(15.0):
                raise AutoProcessError("Timed out while starting the isolated Auto numerical process.")
            ready = self._connection.recv()
            if not bool(ready.get("ok", False)) or ready.get("event") != "ready":
                raise AutoProcessError(str(ready.get("error", "Auto numerical process did not become ready.")))
        except Exception:
            self._force_terminate()
            raise

    @property
    def forced(self) -> bool:
        return bool(self._forced)

    @property
    def alive(self) -> bool:
        try:
            return bool(self._process.is_alive())
        except Exception:
            return False

    def cancel(self) -> None:
        if self._cancel_requested_at is None:
            self._cancel_requested_at = time.monotonic()
        try:
            self._stop_event.set()
        except Exception:
            pass

    def _force_terminate(self) -> None:
        if self._closed:
            return
        self._forced = True
        try:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            if self._process.is_alive() and hasattr(self._process, "kill"):
                self._process.kill()
                self._process.join(timeout=1.0)
        finally:
            try:
                self._connection.close()
            except Exception:
                pass
            self._closed = True

    def request(self, command: Dict[str, Any]) -> Any:
        if self._closed:
            raise AutoCancelledError("The isolated Auto numerical process has already been stopped.")
        if self._cancel_requested_at is not None:
            elapsed = time.monotonic() - self._cancel_requested_at
            if elapsed >= float(self.cancel_grace_seconds):
                self._force_terminate()
                raise AutoCancelledError(
                    f"Current Auto numerical iteration was force-stopped after {self.cancel_grace_seconds:.1f} s."
                )
        try:
            self._connection.send(dict(command))
        except Exception as exc:
            if self._cancel_requested_at is not None:
                self._force_terminate()
                raise AutoCancelledError("Auto numerical process stopped during cancellation.") from exc
            raise AutoProcessError(f"Could not send a request to the Auto numerical process: {exc}") from exc

        while True:
            try:
                if self._connection.poll(0.05):
                    response = self._connection.recv()
                    if not bool(response.get("ok", False)):
                        raise AutoProcessError(str(response.get("error", "Unknown Auto numerical process error.")))
                    return response.get("value")
            except (EOFError, OSError) as exc:
                if self._cancel_requested_at is not None:
                    self._force_terminate()
                    raise AutoCancelledError("Auto numerical process stopped during cancellation.") from exc
                raise AutoProcessError(f"Auto numerical process connection failed: {exc}") from exc

            if self._cancel_requested_at is not None:
                elapsed = time.monotonic() - self._cancel_requested_at
                if elapsed >= float(self.cancel_grace_seconds):
                    self._force_terminate()
                    raise AutoCancelledError(
                        f"Current Auto numerical iteration was force-stopped after {self.cancel_grace_seconds:.1f} s."
                    )
            if not self.alive:
                if self._cancel_requested_at is not None:
                    raise AutoCancelledError("Auto numerical process stopped during cancellation.")
                raise AutoProcessError("Auto numerical process exited unexpectedly.")

    def score_one(self, algorithm: str, params: Dict[str, Any], psf: Optional[PSF]) -> float:
        command = {"op": "score_one", "algorithm": algorithm, "params": dict(params)}
        if self.payload.get("psf") is None and psf is not None:
            command["psf"] = psf
        return float(self.request(command))

    def score_batch(self, algorithm: str, params_list: List[Dict[str, Any]], psf: Optional[PSF]) -> List[float]:
        command = {"op": "score_batch", "algorithm": algorithm, "params_list": [dict(p) for p in params_list]}
        if self.payload.get("psf") is None and psf is not None:
            command["psf"] = psf
        value = self.request(command)
        return [float(v) for v in value]

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.alive and self._cancel_requested_at is None:
                try:
                    self._connection.send({"op": "shutdown"})
                    if self._connection.poll(1.0):
                        self._connection.recv()
                except Exception:
                    pass
            if self.alive:
                self._process.join(timeout=1.0)
            if self.alive:
                self._force_terminate()
        finally:
            try:
                self._connection.close()
            except Exception:
                pass
            self._closed = True


__all__ = [
    "AutoCancelledError",
    "AutoProcessError",
    "AutoNumericalProcessClient",
    "auto_numerical_process_main",
]
