from __future__ import annotations

from ._common import *

class WienerDeconvolution(DeconvolutionAlgorithm):
    name = "Wiener"
    default_params = {
        "K": 0.01, "non_negative": True, "wiener_use_noise_psd": False,
        "wiener_k_scan_enabled": False, "wiener_k_scan_min": 1e-10, "wiener_k_scan_max": 1e-1, "wiener_k_scan_points": 31,
        "use_tv_preconditioning": False, "tv_weight": 0.005, "tv_iterations": 5,
    }

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        k = float(params.get("K", self.default_params["K"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)
        out = wiener_fft_ifft_numpy(image.data, psf.kernel, k, noise_psd=noise_psd, dtype=np.float32)
        out = self._apply_tv_preconditioner(out, tv_enabled, tv_weight, tv_iterations)
        out = self._apply_neural_iteration_denoiser(out, params)
        if non_negative:
            out = np.maximum(out, 0.0)
        boundary_mismatch = convolution_boundary_mismatch(image.data, psf.kernel)
        result = GrayImage(out, "wiener_result")
        psf_meta = getattr(psf, "metadata", {}) or {}
        psf_model = str(psf_meta.get("algorithm_convolution_model", psf_meta.get("convolution_model", "circular_fft")))
        kernel_source = str(psf_meta.get("wiener_kernel_source", "provided_psf"))
        return DeconvolutionResult(result, history=[result], info=f"Wiener K={k}; noise_PSD={noise_psd is not None}; non_negative={non_negative}; PSF_model={psf_model}; kernel_source={kernel_source}; linear_vs_circular_input_mismatch={boundary_mismatch:.6g}")


class TorchBatchWienerDeconvolution(TorchBatchedDeconvolutionMixin, DeconvolutionAlgorithm):
    name = "Torch batch Wiener"
    default_params = {
        "K": 0.01, "non_negative": True, "prefer_cuda": True, "torch_float64": False, "wiener_use_noise_psd": False,
        "wiener_k_scan_enabled": False, "wiener_k_scan_min": 1e-10, "wiener_k_scan_max": 1e-1, "wiener_k_scan_points": 31,
    }

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        result = self.run_batch(image, psf, [params], reference=None, keep_history=True)
        arr = result.infos[0]
        img = GrayImage(arr, name="torch_batch_wiener_result")
        boundary_mismatch = convolution_boundary_mismatch(image.data, psf.kernel)
        psf_meta = getattr(psf, "metadata", {}) or {}
        kernel_source = str(psf_meta.get("wiener_kernel_source", "provided_psf"))
        return DeconvolutionResult(
            img, history=[img],
            info=(f"Torch batch Wiener; device={torch_backend_device(bool(params.get('prefer_cuda', True)))}; "
                  f"kernel_source={kernel_source}; linear_vs_circular_input_mismatch={boundary_mismatch:.6g}")
        )

    def run_batch(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
        device, y, H, _, _ = self._common_batch_setup(image, psf, params_list)
        K = _torch_batch_values([p.get("K", self.default_params["K"]) for p in params_list], self.default_params["K"], "float", device)
        noise_stack = []
        any_noise = False
        for p in params_list:
            psd = normalized_noise_psd_from_image(image, p)
            if psd is None:
                noise_stack.append(np.ones(image.data.shape, dtype=np.float32))
            else:
                any_noise = True
                noise_stack.append(np.asarray(psd, dtype=np.float32))
        noise_tensor = None
        if any_noise:
            noise_tensor = torch.as_tensor(np.asarray(noise_stack), dtype=y.dtype, device=y.device)
        x = _torch_batch_wiener(y, H, K, noise_psd=noise_tensor)
        # Do not apply the generic 1.5 upper clamp here. For small K a Wiener
        # inverse can legitimately have a very large dynamic range; the final
        # GrayImage/display normalization handles it, while scoring clamps only
        # inside the metric function.
        nonneg = _torch_batch_values(
            [p.get("non_negative", self.default_params["non_negative"]) for p in params_list],
            self.default_params["non_negative"],
            "bool",
            device,
        )[:, None, None]
        x = torch.where(nonneg > 0, torch.clamp(x, min=0.0), x)
        scores = self._score_batch_tensor(reference, x) if reference is not None else [float("nan")] * len(params_list)
        arrays = [x[i].detach().cpu().numpy().astype(np.float64) for i in range(len(params_list))]
        return BatchedScores(scores=scores, infos=arrays)

