from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import fftconvolve

from deconv.core.operators import NumpyLinearSameOperator, TorchLinearSameOperator
from deconv.core.runtime import (
    CIRCULAR_FFT,
    LINEAR_SAME,
    GrayImage,
    PSF,
    _torch_batch_values,
    convolution_boundary_mismatch,
    degrade_image,
    degradation_kernel_for_image,
    reconstruction_psf_for_image,
    calculation_psf_for_image,
    wiener_fft_ifft_numpy,
    torch_wiener_filter_np,
    optimize_percentile_range,
    build_intensity_histogram,
    combine_intensity_histograms,
    histogram_quantile,
    histogram_percentile,
    optimize_intensity_levels,
    compute_metrics,
    compute_metrics_batch,
    optimize_psf_floor_and_wiener_k,
    _prepare_psf_candidate_window,
    max_psf_support_for_image,
    zero_outside_psf_rectangle,
    auto_tunable_parameter_names,
)


class LinearOperatorTests(unittest.TestCase):
    def test_numpy_forward_and_adjoint_odd_and_even_kernels(self) -> None:
        rng = np.random.default_rng(123)
        for image_shape in ((17, 19), (18, 20)):
            for kernel_shape in ((3, 5), (4, 6), (7, 7)):
                x = rng.normal(size=image_shape).astype(np.float32)
                y = rng.normal(size=image_shape).astype(np.float32)
                kernel = rng.random(kernel_shape).astype(np.float32)
                kernel /= kernel.sum()
                operator = NumpyLinearSameOperator(kernel, image_shape, dtype=np.float32)

                expected = fftconvolve(x, kernel, mode="same").astype(np.float32)
                np.testing.assert_allclose(operator.forward(x), expected, rtol=2e-5, atol=2e-6)

                lhs = float(np.vdot(operator.forward(x), y))
                rhs = float(np.vdot(x, operator.adjoint(y)))
                self.assertAlmostEqual(lhs, rhs, delta=2e-5 * max(1.0, abs(lhs), abs(rhs)))

    def test_torch_operator_defaults_to_float32(self) -> None:
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch is not installed")
        rng = np.random.default_rng(321)
        shape = (19, 21)
        kernel = rng.random((5, 7)).astype(np.float32)
        kernel /= kernel.sum()
        x = torch.as_tensor(rng.normal(size=(2, *shape)).astype(np.float32))
        operator = TorchLinearSameOperator(kernel, shape, device="cpu")
        result = operator.forward(x)
        self.assertEqual(result.dtype, torch.float32)
        expected = np.stack([fftconvolve(a, kernel, mode="same") for a in x.numpy()]).astype(np.float32)
        np.testing.assert_allclose(result.numpy(), expected, rtol=3e-5, atol=3e-6)

        values = _torch_batch_values([0.1, 0.2], 0.0, "float", "cpu")
        self.assertEqual(values.dtype, torch.float32)


class PercentileOptimizationTests(unittest.TestCase):

    def test_histogram_quantile_and_percentile_are_consistent(self) -> None:
        data = np.linspace(0.0, 1.0, 20000, dtype=np.float32).reshape(100, 200)
        stats = build_intensity_histogram(data, bins=4096)
        q25 = histogram_quantile(stats, 25.0)
        q75 = histogram_quantile(stats, 75.0)
        self.assertAlmostEqual(q25, 0.25, delta=0.002)
        self.assertAlmostEqual(q75, 0.75, delta=0.002)
        self.assertAlmostEqual(histogram_percentile(stats, q25), 25.0, delta=0.2)
        self.assertAlmostEqual(histogram_percentile(stats, q75), 75.0, delta=0.2)

    def test_combined_histogram_without_concatenation(self) -> None:
        a = build_intensity_histogram(np.zeros((64, 64), dtype=np.float32), bins=256)
        b = build_intensity_histogram(np.ones((64, 64), dtype=np.float32), bins=256)
        combined = combine_intensity_histograms([a, b])
        self.assertEqual(combined["count"], 8192)
        self.assertAlmostEqual(histogram_quantile(combined, 25.0), 0.0, delta=0.01)
        self.assertAlmostEqual(histogram_quantile(combined, 75.0), 1.0, delta=0.01)

    def test_fast_intensity_level_optimizer(self) -> None:
        data = np.linspace(0.0, 1.0, 10001, dtype=np.float32)
        stats = build_intensity_histogram(data, bins=4096)
        def score(low: float, high: float) -> float:
            return -((low - 0.01) ** 2 + (high - 0.99) ** 2)
        low, high, score_value, evaluations = optimize_intensity_levels(
            score,
            lambda p: histogram_quantile(stats, p),
            current_low=0.0,
            current_high=1.0,
        )
        self.assertLess(abs(low - 0.01), 0.02)
        self.assertLess(abs(high - 0.99), 0.02)
        self.assertGreater(score_value, -0.001)
        self.assertLessEqual(evaluations, 40)

    def test_percentile_search_refines_to_slider_resolution(self) -> None:
        def score(low: float, high: float) -> float:
            return -((low - 7.3) ** 2 + (high - 98.4) ** 2)

        low, high, best, evaluations = optimize_percentile_range(
            score, current_low=0.0, current_high=97.0
        )
        self.assertAlmostEqual(low, 7.3, delta=0.2)
        self.assertAlmostEqual(high, 98.4, delta=0.2)
        self.assertGreater(best, -0.1)
        self.assertGreater(evaluations, 20)


class WienerFftTests(unittest.TestCase):

    def test_wiener_helper_is_explicit_fft_ifft_formula(self) -> None:
        rng = np.random.default_rng(123)
        image = rng.random((32, 40), dtype=np.float32)
        psf = PSF.gaussian(size=9, sigma=1.6).kernel.astype(np.float32)
        k = np.float32(0.017)
        actual = wiener_fft_ifft_numpy(image, psf, float(k), dtype=np.float32)
        from deconv.core.operators import psf_to_otf_numpy
        from scipy.fft import fft2, ifft2
        H = psf_to_otf_numpy(psf, image.shape, dtype=np.float32)
        expected = np.real(ifft2(np.conj(H) * fft2(image) / (np.abs(H) ** 2 + k))).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

class PsfConsistencyTests(unittest.TestCase):
    def test_repeated_fitting_is_identical_when_settings_are_unchanged(self) -> None:
        source = PSF.gaussian(size=101, sigma=9.0)
        exact = degradation_kernel_for_image(source, (128, 160), max_width=51)
        reconstructed = source.fitted_to_shape((128, 160), max_width=51)
        np.testing.assert_array_equal(exact.kernel, reconstructed.kernel)
        self.assertEqual(exact.metadata.get("convolution_model"), LINEAR_SAME)

    def test_reconstruction_ignores_legacy_snapshot_and_uses_tab2_rectangle(self) -> None:
        source = PSF.gaussian(size=61, sigma=6.0)
        source.metadata.update({
            "calculation_support_height": 19,
            "calculation_support_width": 27,
            "calculation_center_mode": "geometric",
        })
        stale_snapshot = degradation_kernel_for_image(PSF.gaussian(size=31, sigma=2.0), (96, 112), max_width=15)
        prepared_exact_flag = reconstruction_psf_for_image(
            source, stale_snapshot, (96, 112),
            use_exact_degradation_psf=True, max_width=7,
            algorithm_convolution_model=CIRCULAR_FFT,
        )
        prepared_current_flag = reconstruction_psf_for_image(
            source, stale_snapshot, (96, 112),
            use_exact_degradation_psf=False, max_width=55,
            algorithm_convolution_model=CIRCULAR_FFT,
        )
        self.assertEqual(prepared_exact_flag.kernel.shape, (19, 27))
        np.testing.assert_array_equal(prepared_exact_flag.kernel, prepared_current_flag.kernel)
        self.assertEqual(prepared_exact_flag.metadata.get("reconstruction_kernel_source"), "current_tab2_calculation_psf")
        self.assertEqual(prepared_exact_flag.metadata.get("algorithm_convolution_model"), CIRCULAR_FFT)
        self.assertAlmostEqual(float(prepared_exact_flag.kernel.sum()), 1.0, places=7)

    def test_full_psf_preview_mask_zeroes_outside_applied_rectangle(self) -> None:
        source = np.arange(1, 81, dtype=np.float64).reshape(8, 10)
        masked = zero_outside_psf_rectangle(source, center=(3, 6), support_height=4, support_width=5)
        expected = np.zeros_like(source)
        expected[1:5, 4:9] = source[1:5, 4:9]
        np.testing.assert_array_equal(masked, expected)
        compact = PSF.centered_window(masked, (3, 6), 4, 5)
        normalized = PSF.normalize_kernel(compact)
        self.assertAlmostEqual(float(normalized.sum()), 1.0, places=12)

    def test_thresholded_tab2_psf_is_cropped_and_normalized_after_thresholding(self) -> None:
        loaded = PSF.gaussian(size=41, sigma=5.0)
        source = loaded.kernel.copy()
        cutoff = 0.02 * float(source.max())
        thresholded = np.where(source > cutoff, source - cutoff, 0.0)
        current = PSF(
            thresholded, name="thresholded_tab2_psf",
            metadata={
                "calculation_support_height": 17,
                "calculation_support_width": 23,
                "calculation_center_mode": "center_of_mass",
            },
        )
        prepared = calculation_psf_for_image(current, (80, 96))
        self.assertEqual(prepared.kernel.shape, (17, 23))
        self.assertAlmostEqual(float(prepared.kernel.sum()), 1.0, places=7)
        self.assertTrue(np.all(prepared.kernel >= 0.0))
        self.assertEqual(prepared.metadata.get("reconstruction_kernel_source"), "current_tab2_calculation_psf")

    def test_center_of_mass_metadata_keeps_preview_and_operator_aligned(self) -> None:
        kernel = np.zeros((25, 29), dtype=np.float64)
        kernel[8, 20] = 3.0
        kernel[9, 20] = 1.0
        source = PSF(
            kernel,
            metadata={
                "calculation_center_mode": "center_of_mass",
                "calculation_center": (8, 20),
            },
        )
        fitted = source.fitted_to_shape((64, 64), max_width=9)
        self.assertEqual(fitted.metadata.get("source_psf_center"), (8, 20))
        self.assertEqual(np.unravel_index(int(np.argmax(fitted.kernel)), fitted.kernel.shape), (4, 4))
        self.assertAlmostEqual(float(fitted.kernel.sum()), 1.0, places=12)

    def test_manual_psf_center_selects_requested_source_window(self) -> None:
        kernel = np.zeros((19, 23), dtype=np.float64)
        kernel[6, 15] = 4.0
        kernel[7, 15] = 1.0
        source = PSF(
            kernel,
            name="off_axis_psf",
            metadata={
                "calculation_center_mode": "manual",
                "calculation_center": (6, 15),
            },
        )
        fitted = source.fitted_to_shape((64, 64), max_width=7)
        self.assertEqual(fitted.kernel.shape, (7, 7))
        self.assertEqual(np.unravel_index(int(np.argmax(fitted.kernel)), fitted.kernel.shape), (3, 3))
        self.assertEqual(fitted.metadata.get("source_psf_center"), (6, 15))
        self.assertEqual(fitted.metadata.get("source_calculation_center_mode"), "manual")
        self.assertEqual(fitted.metadata.get("calculation_center"), (3, 3))
        refitted = fitted.fitted_to_shape((64, 64), max_width=7)
        np.testing.assert_array_equal(refitted.kernel, fitted.kernel)

    def test_automatic_support_selection_tracks_off_axis_signal(self) -> None:
        yy, xx = np.indices((81, 91))
        raw = 0.015 + np.exp(-((xx - 63.0) ** 2 + (yy - 28.0) ** 2) / (2.0 * 4.0 ** 2))
        selection = PSF.automatic_support_selection(raw)
        cy, cx = selection["center"]
        self.assertLessEqual(abs(cy - 28), 1)
        self.assertLessEqual(abs(cx - 63), 1)
        self.assertGreater(selection["width"], 9)
        self.assertLess(selection["width"], min(raw.shape))
        self.assertGreater(selection["floor_fraction"], 0.0)


    def test_automatic_support_selection_is_rectangular_and_uses_one_percent_peak(self) -> None:
        yy, xx = np.indices((101, 121))
        raw = 0.002 + np.exp(-((xx - 76.0) ** 2) / (2.0 * 9.0 ** 2) - ((yy - 33.0) ** 2) / (2.0 * 2.0 ** 2))
        selection = PSF.automatic_support_selection(raw)
        self.assertGreater(selection["width"], selection["height"])
        self.assertGreaterEqual(selection["floor_fraction"], 0.01 - 1e-12)
        self.assertAlmostEqual(selection["peak_fraction"], 1e-2)

    def test_rectangular_fitted_psf_uses_selected_shape_and_normalizes(self) -> None:
        raw = np.zeros((51, 73), dtype=np.float64)
        raw[20:27, 31:48] = 1.0
        source = PSF(raw, metadata={
            "calculation_center_mode": "manual",
            "calculation_center": (23, 39),
            "calculation_support_height": 11,
            "calculation_support_width": 25,
        })
        fitted = source.fitted_to_shape((80, 90), max_width=25)
        self.assertEqual(fitted.kernel.shape, (11, 25))
        self.assertAlmostEqual(float(fitted.kernel.sum()), 1.0, places=12)
        self.assertEqual(fitted.metadata["fitted_support"], (11, 25))

    def test_even_rectangular_selection_preserves_exact_shape(self) -> None:
        raw = np.zeros((40, 60), dtype=np.float64)
        raw[14:26, 18:42] = 1.0
        source = PSF(raw, metadata={
            "calculation_center_mode": "manual",
            "calculation_center": (20, 30),
            "calculation_support_height": 12,
            "calculation_support_width": 24,
        })
        fitted = source.fitted_to_shape((40, 60), max_width=24)
        self.assertEqual(fitted.kernel.shape, (12, 24))
        self.assertAlmostEqual(float(fitted.kernel.sum()), 1.0, places=12)
        self.assertEqual(max_psf_support_for_image((40, 60)), 40)

    def test_optimizer_candidate_is_cropped_to_rectangle_and_normalized(self) -> None:
        raw = np.zeros((41, 61), dtype=np.float64)
        raw[15:24, 20:45] = 0.2
        raw[19, 32] = 1.0
        crop, meta = _prepare_psf_candidate_window(
            raw, center=(19, 32), support_height=13, support_width=31, floor_fraction=0.05
        )
        self.assertEqual(crop.shape, (13, 31))
        self.assertAlmostEqual(float(crop.sum()), 1.0, places=12)
        self.assertEqual(meta["support_height"], 13)
        self.assertEqual(meta["support_width"], 31)
        self.assertAlmostEqual(meta["normalized_sum"], 1.0, places=12)

    def test_cropped_psf_is_normalized_after_floor(self) -> None:
        yy, xx = np.indices((41, 45))
        raw = 0.01 + np.exp(-((xx - 29.0) ** 2 + (yy - 17.0) ** 2) / (2.0 * 3.0 ** 2))
        cutoff = 0.02 * float(raw.max())
        thresholded = np.maximum(raw - cutoff, 0.0)
        source = PSF(
            thresholded,
            metadata={"calculation_center_mode": "manual", "calculation_center": (17, 29)},
        )
        fitted = source.fitted_to_shape((96, 96), max_width=19)
        self.assertAlmostEqual(float(fitted.kernel.sum()), 1.0, places=12)
        self.assertEqual(fitted.kernel.shape, (19, 19))

    def test_joint_psf_floor_wiener_optimizer_returns_finite_parameters(self) -> None:
        reference = GrayImage.synthetic(64, 64, padding=8)
        true_psf = PSF.gaussian(11, 1.8)
        measured = degrade_image(reference, true_psf, noise_sigma=0.002)
        raw = np.pad(true_psf.kernel, ((8, 8), (8, 8)), constant_values=0.001)
        result = optimize_psf_floor_and_wiener_k(
            measured.data, raw, center=(13, 13), support_width=20, support_height=14,
            reference=reference.data, current_k=0.01, current_floor=0.0, max_preview_side=64,
        )
        self.assertTrue(np.isfinite(result["cost"]))
        self.assertGreaterEqual(result["floor_fraction"], 0.0)
        self.assertLessEqual(result["floor_fraction"], 0.5)
        self.assertGreater(result["K"], 0.0)
        self.assertEqual(result["criterion"], "MSE")
        self.assertEqual(result["support_height"], 14)
        self.assertEqual(result["support_width"], 20)
        self.assertEqual(result["candidate_psf"]["support_height"], 14)
        self.assertEqual(result["candidate_psf"]["support_width"], 20)
        self.assertTrue(
            abs(float(result["floor_fraction"]) - float(result["initial_floor_fraction"])) > 1e-10
            or abs(np.log10(float(result["K"]) / float(result["initial_K"]))) > 1e-6
        )
        self.assertAlmostEqual(result["candidate_psf"]["normalized_sum"], 1.0, places=6)
        self.assertLessEqual(result["cost"], result["initial_cost"] + 1e-10)

    def test_joint_psf_floor_wiener_optimizer_supports_gcv_without_reference(self) -> None:
        reference = GrayImage.synthetic(48, 52, padding=6)
        true_psf = PSF.gaussian(9, 1.5)
        measured = degrade_image(reference, true_psf, noise_sigma=0.003)
        raw = np.pad(true_psf.kernel, ((6, 6), (6, 6)), constant_values=0.0005)
        result = optimize_psf_floor_and_wiener_k(
            measured.data, raw, center=(10, 10), support_width=17, support_height=11,
            reference=None, current_k=0.01, current_floor=0.0, max_preview_side=64,
        )
        self.assertTrue(np.isfinite(result["cost"]))
        self.assertGreater(result["K"], 0.0)
        self.assertEqual(result["criterion"], "Constrained PSF background + conditional Wiener GCV")
        self.assertAlmostEqual(result["candidate_psf"]["normalized_sum"], 1.0, places=6)
        self.assertLessEqual(result["cost"], result["initial_cost"] + 1e-10)


    def test_no_reference_joint_optimizer_rejects_collapsed_high_floor(self) -> None:
        reference = GrayImage.synthetic(96, 96, padding=12)
        true_psf = PSF.gaussian(19, 2.8)
        measured = degrade_image(reference, true_psf, noise_sigma=0.008)
        rng = np.random.default_rng(404)
        raw = np.pad(true_psf.kernel, ((16, 16), (20, 20)), constant_values=0.0025)
        raw = np.maximum(raw + rng.normal(0.0, 0.001, raw.shape), 0.0)
        result = optimize_psf_floor_and_wiener_k(
            measured.data, raw, center=(25, 29), support_width=39, support_height=35,
            reference=None, current_k=0.01, current_floor=0.0, max_preview_side=96,
        )
        lo, hi = result["floor_search_bounds"]
        self.assertGreaterEqual(result["floor_fraction"], lo - 1e-12)
        self.assertLessEqual(result["floor_fraction"], hi + 1e-12)
        self.assertLessEqual(hi, 0.25 + 1e-12)
        self.assertGreaterEqual(result["candidate_psf"]["retained_mass_fraction"], 0.05)
        self.assertGreater(result["candidate_psf"]["effective_pixels"], 2.5)
        self.assertGreater(result["candidate_psf"]["nonzero_pixels"], 4)
        self.assertIn("conditional_gcv", result["criterion_components"])

    def test_auto_freezes_parameters_of_disabled_processing_stages(self) -> None:
        active = [
            "iterations", "K", "begin_with_wiener", "wiener_use_noise_psd",
            "use_tv_preconditioning", "tv_weight", "tv_iterations",
            "neural_denoiser_mode", "denoiser_type", "neural_denoiser_strength",
            "rosen_relax_to_one", "rosen_relax_factor",
        ]
        disabled = {
            "begin_with_wiener": False,
            "wiener_use_noise_psd": False,
            "use_tv_preconditioning": False,
            "neural_denoiser_mode": "Off",
            "denoiser_type": "Gaussian",
            "rosen_relax_to_one": False,
        }
        allowed = set(auto_tunable_parameter_names("Richardson-Lucy", active, disabled))
        self.assertIn("iterations", allowed)
        self.assertNotIn("K", allowed)
        self.assertNotIn("begin_with_wiener", allowed)
        self.assertNotIn("wiener_use_noise_psd", allowed)
        self.assertNotIn("tv_weight", allowed)
        self.assertNotIn("tv_iterations", allowed)
        self.assertNotIn("neural_denoiser_mode", allowed)
        self.assertNotIn("denoiser_type", allowed)
        self.assertNotIn("neural_denoiser_strength", allowed)
        self.assertNotIn("rosen_relax_factor", allowed)

        enabled = dict(disabled)
        enabled.update({
            "begin_with_wiener": True,
            "use_tv_preconditioning": True,
            "neural_denoiser_mode": "After each iteration",
            "rosen_relax_to_one": True,
        })
        allowed = set(auto_tunable_parameter_names("Richardson-Lucy-Rosen", active, enabled))
        self.assertIn("K", allowed)
        self.assertIn("tv_weight", allowed)
        self.assertIn("tv_iterations", allowed)
        self.assertIn("denoiser_type", allowed)
        self.assertIn("neural_denoiser_strength", allowed)
        self.assertIn("rosen_relax_factor", allowed)
        # Feature activation controls themselves remain frozen.
        self.assertNotIn("begin_with_wiener", allowed)
        self.assertNotIn("use_tv_preconditioning", allowed)
        self.assertNotIn("neural_denoiser_mode", allowed)

        direct = set(auto_tunable_parameter_names("Wiener", active, disabled))
        self.assertIn("K", direct)

    def test_torch_wiener_initializer_ignores_deprecated_absolute_output(self) -> None:
        rng = np.random.default_rng(88)
        data = rng.normal(size=(32, 36)).astype(np.float32)
        kernel = PSF.gaussian(9, 1.8).kernel
        psd = np.linspace(0.5, 1.5, data.size, dtype=np.float32).reshape(data.shape)
        standard = torch_wiener_filter_np(
            data, kernel, 0.03, device="cpu", torch_float64=False,
            noise_psd=psd, absolute_output=False,
        )
        deprecated_mode = torch_wiener_filter_np(
            data, kernel, 0.03, device="cpu", torch_float64=False,
            noise_psd=psd, absolute_output=True,
        )
        self.assertEqual(standard.dtype, np.float32)
        self.assertEqual(deprecated_mode.dtype, np.float32)
        self.assertTrue(np.isfinite(standard).all())
        np.testing.assert_allclose(standard, deprecated_mode, rtol=0.0, atol=0.0)
        self.assertLess(float(np.min(standard)), 0.0)

    def test_degradation_records_linear_model(self) -> None:
        image = GrayImage.synthetic(64, 64, padding=8)
        psf = degradation_kernel_for_image(PSF.gaussian(13, 2.0), image.data.shape, max_width=13)
        degraded = degrade_image(image, psf, noise_sigma=0.0)
        self.assertEqual(degraded.metadata.get("convolution_model"), LINEAR_SAME)
        self.assertEqual(tuple(degraded.metadata.get("degradation_psf_shape")), psf.kernel.shape)

    def test_boundary_mismatch_diagnoses_wraparound_not_psf_identity(self) -> None:
        psf = PSF.gaussian(15, 2.5)
        framed = np.zeros((64, 64), dtype=np.float32)
        framed[20:44, 20:44] = 1.0
        edge_filled = np.ones((64, 64), dtype=np.float32)
        self.assertLess(convolution_boundary_mismatch(framed, psf.kernel), 1e-5)
        self.assertGreater(convolution_boundary_mismatch(edge_filled, psf.kernel), 1e-2)


class BatchedMetricsTests(unittest.TestCase):
    def test_batch_metrics_match_scalar_metrics(self) -> None:
        rng = np.random.default_rng(2026)
        reference = GrayImage.synthetic(64, 64, padding=8)
        psf = degradation_kernel_for_image(PSF.gaussian(9, 1.6), reference.data.shape, max_width=9)
        measured = degrade_image(reference, psf, noise_sigma=0.01)
        history = []
        for sigma in (0.0, 0.01, 0.03, 0.06):
            data = np.clip(reference.data + sigma * rng.normal(size=reference.data.shape), 0.0, 1.0)
            history.append(GrayImage(data, name=f"frame_{sigma}", metadata={"_preserve_intensity": True}))

        diagnostics = {}
        batched = compute_metrics_batch(
            reference,
            history,
            allow_reference_metrics=True,
            roi_source=measured,
            measured=measured,
            psfs=[psf] * len(history),
            prefer_cuda=False,
            diagnostics=diagnostics,
        )
        scalar = [compute_metrics(reference, frame, True, measured, measured, psf) for frame in history]
        self.assertEqual(diagnostics.get("batches"), 1)
        self.assertEqual(diagnostics.get("batch_size"), len(history))
        for batch_item, scalar_item in zip(batched, scalar):
            for key in (
                "TV", "NTV", "RELATIVE_REBLUR_RESIDUAL", "RELATIVE_INTENSITY_ERROR",
                "RESIDUAL_WHITENESS", "NO_REFERENCE_COST", "PSNR", "SSIM",
            ):
                self.assertAlmostEqual(batch_item[key], scalar_item[key], delta=3e-5, msg=key)

    def test_batch_metrics_accept_per_iteration_psfs(self) -> None:
        reference = GrayImage.synthetic(48, 52, padding=6)
        measured_psf = degradation_kernel_for_image(PSF.gaussian(7, 1.4), reference.data.shape, max_width=7)
        measured = degrade_image(reference, measured_psf, noise_sigma=0.0)
        history = [
            GrayImage(np.clip(reference.data * factor, 0.0, 1.0), metadata={"_preserve_intensity": True})
            for factor in (0.90, 1.00, 1.08)
        ]
        psfs = [
            degradation_kernel_for_image(PSF.gaussian(7, sigma), reference.data.shape, max_width=7)
            for sigma in (1.2, 1.4, 1.8)
        ]
        diagnostics = {}
        batched = compute_metrics_batch(
            None,
            history,
            allow_reference_metrics=False,
            roi_source=measured,
            measured=measured,
            psfs=psfs,
            prefer_cuda=False,
            diagnostics=diagnostics,
        )
        scalar = [compute_metrics(None, frame, False, measured, measured, psf) for frame, psf in zip(history, psfs)]
        self.assertEqual(len(batched), len(history))
        self.assertEqual(diagnostics.get("psf_groups"), 1)
        for batch_item, scalar_item in zip(batched, scalar):
            for key in ("TV", "NTV", "RELATIVE_REBLUR_RESIDUAL", "RESIDUAL_WHITENESS", "NO_REFERENCE_COST"):
                self.assertAlmostEqual(batch_item[key], scalar_item[key], delta=3e-5, msg=key)


if __name__ == "__main__":
    unittest.main()


def test_psf_fitted_to_shape_respects_geometric_center_mode():
    from deconv.core.runtime import PSF
    import numpy as np
    arr = np.zeros(( nine := 9, nine), dtype=np.float64)
    arr[1, 1] = 1.0
    psf_com = PSF(arr, metadata={"calculation_center_mode": "center_of_mass"})
    psf_geo = PSF(arr, metadata={"calculation_center_mode": "geometric"})
    com = psf_com.fitted_to_shape((9, 9), max_width=5)
    geo = psf_geo.fitted_to_shape((9, 9), max_width=5)
    assert np.unravel_index(np.argmax(com.kernel), com.kernel.shape) == (2, 2)
    assert float(geo.kernel.sum()) > 0.0
    assert geo.metadata["calculation_center_mode"] == "geometric"
