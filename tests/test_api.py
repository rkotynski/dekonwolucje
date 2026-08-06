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


def test_api_exposes_all_registered_algorithms_and_gui_psf_generators():
    from deconv.api import (
        available_psf_generators,
        generate_gaussian_psf,
        generate_high_frequency_psf,
        generate_lens_incoherent_psf,
    )
    from deconv.algorithms.registry import AlgorithmRegistry

    assert set(available_algorithms()) == set(AlgorithmRegistry().names())
    assert available_psf_generators() == ("gaussian", "motion", "high_frequency", "lens_incoherent")
    psfs = (
        generate_gaussian_psf(size=9, sigma=1.5),
        generate_motion_psf(size=9, angle_deg=25.0),
        generate_high_frequency_psf(size=9, frequency=2.0, sigma=2.0),
        generate_lens_incoherent_psf(size=9, diffraction_grid_size=19),
    )
    for psf in psfs:
        assert psf.kernel.ndim == 2
        assert np.isclose(psf.kernel.sum(), 1.0)
        assert np.all(psf.kernel >= 0.0)


def test_cpu_convenience_wrappers_run():
    from deconv.api import (
        block_kaczmarz_filter,
        landweber_filter,
        landweber_wiener_preconditioned_filter,
        richardson_lucy_filter,
        richardson_lucy_rosen_filter,
        richardson_lucy_wiener_filter,
    )
    reference = generate_test_image(width=48, height=40)
    psf = generate_motion_psf(size=5, angle_deg=20.0)
    disturbed = disturb_image(reference, psf, noise_sigma=0.0, seed=1)
    calls = (
        lambda: richardson_lucy_filter(disturbed, psf, iterations=1),
        lambda: richardson_lucy_wiener_filter(disturbed, psf, iterations=1),
        lambda: richardson_lucy_rosen_filter(disturbed, psf, iterations=1),
        lambda: landweber_filter(disturbed, psf, iterations=1),
        lambda: landweber_wiener_preconditioned_filter(disturbed, psf, iterations=1),
        lambda: block_kaczmarz_filter(disturbed, psf, iterations=1, block_size=16),
    )
    for call in calls:
        result = call()
        assert result.image.data.shape == reference.data.shape
        assert np.all(np.isfinite(result.image.data))


def test_auto_api_tunes_wiener_and_can_run_best_result():
    from deconv.api import AutoTuneOptions, auto_tune_parameters

    reference = generate_test_image(width=40, height=32)
    psf = generate_motion_psf(size=7, angle_deg=30.0)
    disturbed = disturb_image(reference, psf, noise_sigma=0.01, seed=12)
    tuning = auto_tune_parameters(
        disturbed,
        psf,
        algorithm="Wiener",
        reference=reference,
        params={"K": 0.1},
        auto_options=AutoTuneOptions(
            strategy="quadratic",
            use_torch_equivalent=False,
        ),
        run_best=True,
    )
    assert tuning.best_params["K"] > 0.0
    assert np.isfinite(tuning.best_score)
    assert tuning.best_score >= tuning.initial_score - 1e-9
    assert tuning.evaluations > 1
    assert tuning.deconvolution_result is not None
    assert tuning.deconvolution_result.image.data.shape == reference.data.shape


def test_auto_api_keeps_disabled_stage_parameters_unchanged():
    from deconv.api import auto_tune_parameters

    reference = generate_test_image(width=28, height=24)
    psf = generate_motion_psf(size=5, angle_deg=20.0)
    disturbed = disturb_image(reference, psf, noise_sigma=0.0, seed=3)
    tuning = auto_tune_parameters(
        disturbed,
        psf,
        algorithm="Richardson-Lucy",
        reference=reference,
        params={
            "iterations": 2,
            "begin_with_wiener": False,
            "K": 0.123,
            "use_tv_preconditioning": False,
            "tv_weight": 0.777,
        },
        auto_options={
            "strategy": "quadratic",
            "passes": 1,
            "use_torch_equivalent": False,
        },
    )
    assert tuning.best_params["begin_with_wiener"] is False
    assert tuning.best_params["K"] == 0.123
    assert tuning.best_params["use_tv_preconditioning"] is False
    assert tuning.best_params["tv_weight"] == 0.777


def test_auto_api_strategies_and_example(tmp_path: Path):
    from deconv.api import available_auto_strategies

    assert available_auto_strategies() == ("quadratic", "full_batched")
    root = Path(__file__).parents[1]
    output = tmp_path / "auto_example"
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [sys.executable, str(root / "examples" / "auto_richardson_lucy_motion.py"), "--output-dir", str(output)],
        cwd=root,
        env=env,
        check=True,
        timeout=120,
    )
    for name in ("reference.png", "motion_psf.png", "disturbed.png", "restored_auto_rl.png", "comparison.png"):
        assert (output / name).stat().st_size > 0
