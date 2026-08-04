from __future__ import annotations

from ._common import *

class LandweberDeconvolution(DeconvolutionAlgorithm):
    """Iterative Landweber deconvolution using gradient-descent updates."""

    name = "Landweber"
    default_params = {"iterations": 50, "step": 0.8, "K": 0.01, "non_negative": True, "begin_with_wiener": False,
                      "wiener_use_noise_psd": False,
                      "use_tv_preconditioning": False, "tv_weight": 0.005, "tv_iterations": 5}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        step = float(params.get("step", self.default_params["step"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)

        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        k_init = float(params.get("K", self.default_params["K"]))
        estimate = wiener_fft_ifft_numpy(
            image.data, psf.kernel, k_init, noise_psd=noise_psd
        ) if begin_with_wiener else image.data.copy()
        if non_negative:
            estimate = np.maximum(estimate, 0.0)
        operator = NumpyLinearSameOperator(psf.kernel, image.data.shape, dtype=np.float32)
        history: List[GrayImage] = []

        for i in range(iterations):
            blurred_estimate = operator.forward(estimate)
            residual = np.asarray(image.data, dtype=np.float32) - blurred_estimate
            gradient = operator.adjoint(residual)
            estimate = estimate + step * gradient
            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)
            history.append(GrayImage(estimate.copy(), name=f"landweber_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break

        return DeconvolutionResult(
            history[-1],
            history=history,
            info=f"Landweber iterations={iterations}; step={step}; non_negative={non_negative}; begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, weight={tv_weight}, tv_iter={tv_iterations}",
        )

class LandweberWienerPreconditionedDeconvolution(DeconvolutionAlgorithm):
    """Landweber iteration with a Wiener-type spectral preconditioner."""

    name = "Landweber Wiener-preconditioned"
    default_params = {"iterations": 50, "step": 0.8, "K": 0.01, "non_negative": True, "begin_with_wiener": False,
                      "wiener_use_noise_psd": False,
                      "use_tv_preconditioning": False, "tv_weight": 0.005, "tv_iterations": 5}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        step = float(params.get("step", self.default_params["step"]))
        k = float(params.get("K", self.default_params["K"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)

        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        estimate = wiener_fft_ifft_numpy(
            image.data, psf.kernel, k, noise_psd=noise_psd
        ) if begin_with_wiener else image.data.copy()
        if non_negative:
            estimate = np.maximum(estimate, 0.0)
        operator = NumpyLinearSameOperator(psf.kernel, image.data.shape, dtype=np.float32)
        H = psf_to_otf_numpy(psf.kernel, image.data.shape, dtype=np.float32)
        N = np.ones(image.data.shape, dtype=np.float32) if noise_psd is None else np.asarray(noise_psd, dtype=np.float32)
        history: List[GrayImage] = []

        for i in range(iterations):
            blurred_estimate = operator.forward(estimate)
            residual = np.asarray(image.data, dtype=np.float32) - blurred_estimate
            gradient = operator.adjoint(residual)
            preconditioned_gradient = np.real(ifft2(fft2(gradient) / (np.abs(H) ** 2 + np.float32(k) * N)))
            estimate = estimate + step * preconditioned_gradient
            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)
            history.append(GrayImage(estimate.copy(), name=f"landweber_wiener_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break

        return DeconvolutionResult(
            history[-1],
            history=history,
            info=f"Landweber Wiener-preconditioned iterations={iterations}; step={step}; K={k}; non_negative={non_negative}; begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, weight={tv_weight}, tv_iter={tv_iterations}",
        )

class TorchBatchLandweberDeconvolution(TorchBatchedDeconvolutionMixin, DeconvolutionAlgorithm):
    name = "Torch batch Landweber"
    default_params = {"iterations": 50, "step": 0.8, "K": 0.01, "begin_with_wiener": False, "non_negative": True,
                      "wiener_use_noise_psd": False,
                      "prefer_cuda": True, "torch_float64": False}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        params = dict(params)
        device, y, H, _, _ = self._common_batch_setup(image, psf, [params])
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        step = float(params.get("step", self.default_params["step"]))
        x = self._initial_estimate_batch(y, H, [params], default_k=float(params.get("K", 0.01)), image=image)
        history: List[GrayImage] = []
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        for i in range(iterations):
            residual = y - operator.forward(x)
            correction = operator.adjoint(residual)
            x = x + step * correction
            x = _torch_batch_tv_step_np_batch(x, [params])
            x = _torch_batch_neural_step_np_batch(x, [params])
            x = self._finalize_batch(x, [params])
            history.append(GrayImage(x[0].detach().cpu().numpy().astype(np.float64), name=f"torch_batch_landweber_iteration_{i+1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
        return DeconvolutionResult(history[-1], history=history, info=f"Torch batch Landweber iterations={iterations}; step={step}; device={device}; convolution=linear same")

    def run_batch(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
        device, y, H, _, _ = self._common_batch_setup(image, psf, params_list)
        max_iter = max(int(p.get("iterations", self.default_params["iterations"])) for p in params_list)
        iters = _torch_batch_values([p.get("iterations", self.default_params["iterations"]) for p in params_list], self.default_params["iterations"], "int", device)[:, None, None]
        steps = _torch_batch_values([p.get("step", self.default_params["step"]) for p in params_list], self.default_params["step"], "float", device)[:, None, None]
        x = self._initial_estimate_batch(y, H, params_list, default_k=0.01, image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        best = [float("-inf")] * len(params_list)
        for i in range(max_iter):
            if self._stop_requested(params_list[0]):
                break
            active = (iters > i).float()
            residual = y - operator.forward(x)
            x_new = x + steps * operator.adjoint(residual)
            x_new = _torch_batch_tv_step_np_batch(x_new, params_list)
            x_new = _torch_batch_neural_step_np_batch(x_new, params_list)
            x_new = self._finalize_batch(x_new, params_list)
            x = active * x_new + (1.0 - active) * x
            if reference is not None:
                scores = self._score_batch_tensor(reference, x)
                for j, s in enumerate(scores):
                    if i < int(params_list[j].get("iterations", self.default_params["iterations"])):
                        best[j] = max(best[j], s)
        if reference is None:
            best = [float("nan")] * len(params_list)
        arrays = [x[i].detach().cpu().numpy().astype(np.float64) for i in range(len(params_list))]
        return BatchedScores(scores=best, infos=arrays)

