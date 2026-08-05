from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from deconv.api import (
    as_gray_image,
    available_algorithms,
    default_parameters,
    disturb_image,
    generate_motion_psf,
    generate_test_image,
    run_deconvolution,
    wiener_filter,
)


def test_public_api_runs_without_importing_qt():
    code = "import sys; import deconv.api; assert not any(name.startswith('PyQt5') for name in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_generated_wiener_problem_is_reproducible_and_finite():
    reference = generate_test_image(width=96, height=64)
    psf = generate_motion_psf(size=11, angle_deg=35.0)
    disturbed_a = disturb_image(reference, psf, noise_sigma=0.01, seed=123)
    disturbed_b = disturb_image(reference, psf, noise_sigma=0.01, seed=123)
    assert np.array_equal(disturbed_a.data, disturbed_b.data)

    result = wiener_filter(disturbed_a, psf, K=2e-3)
    assert result.image.data.shape == reference.data.shape
    assert np.all(np.isfinite(result.image.data))
    assert result.history and result.history[0].data.shape == reference.data.shape


def test_general_registry_api_and_numpy_inputs():
    assert "Wiener" in available_algorithms()
    defaults = default_parameters("richardson-lucy")
    assert defaults["iterations"] > 0

    image = np.zeros((32, 40), dtype=np.float32)
    image[10:22, 14:26] = 1.0
    psf = generate_motion_psf(size=7, angle_deg=20.0).kernel
    disturbed = disturb_image(image, psf, noise_sigma=0.0)
    result = run_deconvolution(
        disturbed.data,
        psf,
        algorithm="Richardson-Lucy",
        iterations=2,
        epsilon=1e-8,
    )
    assert result.image.data.shape == image.shape
    assert len(result.history) == 2


def test_array_input_validation_preserves_unit_range():
    arr = np.full((8, 9), 0.25, dtype=np.float32)
    model = as_gray_image(arr)
    assert np.allclose(model.data, 0.25)


def test_standalone_wiener_example(tmp_path: Path):
    root = Path(__file__).parents[1]
    output = tmp_path / "example"
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, str(root / "examples" / "wiener_motion_blur.py"), "--output-dir", str(output)],
        cwd=root,
        env=env,
        check=True,
        timeout=60,
    )
    for name in ("reference.png", "motion_psf.png", "disturbed.png", "restored_wiener.png", "comparison.png"):
        assert (output / name).stat().st_size > 0
