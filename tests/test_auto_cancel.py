from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from deconv.core.runtime import TORCH_AVAILABLE, GrayImage, PSF
from deconv.optim.auto_process import AutoCancelledError, AutoNumericalProcessClient


def _payload():
    image = GrayImage(
        np.zeros((16, 16), dtype=np.float64),
        name="measured",
        metadata={"measured_input": True},
    )
    return {"reference": None, "degraded": image, "allow_reference": False}


def test_isolated_auto_process_can_be_force_stopped_after_grace_period():
    client = AutoNumericalProcessClient(_payload(), cancel_grace_seconds=0.25)
    timer = threading.Timer(0.05, client.cancel)
    timer.start()
    started = time.monotonic()
    try:
        try:
            client.request({"op": "sleep", "seconds": 10.0, "ignore_cancel": True})
            raise AssertionError("The uncooperative request should have been force-stopped.")
        except AutoCancelledError:
            elapsed = time.monotonic() - started
            assert elapsed < 2.0
            assert client.forced
    finally:
        timer.cancel()
        client.close()


def test_isolated_auto_process_stops_cooperatively_before_hard_limit():
    client = AutoNumericalProcessClient(_payload(), cancel_grace_seconds=3.0)
    timer = threading.Timer(0.05, client.cancel)
    timer.start()
    started = time.monotonic()
    try:
        result = client.request({"op": "sleep", "seconds": 10.0, "ignore_cancel": False})
        elapsed = time.monotonic() - started
        assert result == "slept"
        assert elapsed < 3.0
        assert not client.forced
    finally:
        timer.cancel()
        client.close()


def test_isolated_auto_process_scores_a_wiener_candidate():
    rng = np.random.default_rng(4)
    image = GrayImage(rng.random((32, 32)), name="measured", metadata={"measured_input": True})
    kernel = np.array([[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]], dtype=np.float64)
    psf = PSF(kernel, name="test_psf")
    payload = {"reference": None, "degraded": image, "psf": psf, "allow_reference": False}
    client = AutoNumericalProcessClient(payload, cancel_grace_seconds=5.0)
    try:
        score = client.score_one("Wiener", {"K": 0.01, "wiener_use_noise_psd": False}, psf)
        assert np.isfinite(score)
    finally:
        client.close()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_isolated_auto_process_scores_a_small_batched_rl_search():
    reference_data = np.zeros((24, 24), dtype=np.float64)
    reference_data[8:16, 9:15] = 1.0
    reference = GrayImage(reference_data, name="reference")
    kernel = np.array([[0.0, 0.1, 0.0], [0.1, 0.6, 0.1], [0.0, 0.1, 0.0]], dtype=np.float64)
    psf = PSF(kernel, name="test_psf")
    # A simple circular blur is sufficient for an executor integration test.
    H = np.fft.fft2(np.fft.ifftshift(np.pad(psf.kernel, ((10, 11), (10, 11)))))
    degraded_data = np.real(np.fft.ifft2(np.fft.fft2(reference.data) * H))
    degraded = GrayImage(degraded_data, name="degraded")
    payload = {"reference": reference, "degraded": degraded, "psf": psf, "allow_reference": True}
    client = AutoNumericalProcessClient(payload, cancel_grace_seconds=5.0)
    try:
        candidates = [
            {"iterations": 2, "epsilon": 1e-8, "prefer_cuda": False, "non_negative": True},
            {"iterations": 3, "epsilon": 1e-8, "prefer_cuda": False, "non_negative": True},
        ]
        scores = client.score_batch("Torch batch Richardson-Lucy", candidates, psf)
        assert len(scores) == 2
        assert np.isfinite(np.asarray(scores)).all()
    finally:
        client.close()
