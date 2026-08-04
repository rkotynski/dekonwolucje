from __future__ import annotations

from ._common import *

class RichardsonLucyDeconvolution(DeconvolutionAlgorithm):
    name = "Richardson-Lucy"
    default_params = {"iterations": 20, "epsilon": 1e-8, "K": 0.01, "non_negative": True, "begin_with_wiener": False,
                      "wiener_use_noise_psd": False,
                      "use_tv_preconditioning": False, "tv_weight": 0.005, "tv_iterations": 5}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps = float(params.get("epsilon", self.default_params["epsilon"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)
        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        k_init = float(params.get("K", self.default_params["K"]))
        estimate = wiener_fft_ifft_numpy(
            image.data, psf.kernel, k_init, noise_psd=noise_psd
        ) if begin_with_wiener else np.full_like(image.data, 0.5)
        if non_negative:
            estimate = np.maximum(estimate, 0.0)
        operator = NumpyLinearSameOperator(psf.kernel, image.data.shape, dtype=np.float32)
        measured = np.asarray(image.data, dtype=np.float32)
        history: List[GrayImage] = []
        for i in range(iterations):
            conv = operator.forward(estimate) + eps
            relative_blur = measured / conv
            estimate *= operator.adjoint(relative_blur)
            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)
            history.append(GrayImage(estimate.copy(), name=f"rl_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
        neural_mode = DeconvolutionAlgorithm._neural_denoiser_mode(params)
        return DeconvolutionResult(history[-1], history=history, info=f"RL iterations={iterations}; non_negative={non_negative}; begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, weight={tv_weight}, tv_iter={tv_iterations}; neural={neural_mode}")

class RichardsonLucyWienerDeconvolution(DeconvolutionAlgorithm):
    """Richardson-Lucy iterations with a Wiener filtering step after each RL update."""

    name = "Richardson-Lucy-Wiener"
    default_params = {"iterations": 20, "epsilon": 1e-8, "K": 0.01, "non_negative": True, "begin_with_wiener": False,
                      "wiener_use_noise_psd": False,
                      "use_tv_preconditioning": False, "tv_weight": 0.005, "tv_iterations": 5}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps = float(params.get("epsilon", self.default_params["epsilon"]))
        k = float(params.get("K", self.default_params["K"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)
        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        estimate = wiener_fft_ifft_numpy(
            image.data, psf.kernel, k, noise_psd=noise_psd
        ) if begin_with_wiener else np.full_like(image.data, 0.5)
        if non_negative:
            estimate = np.maximum(estimate, 0.0)
        operator = NumpyLinearSameOperator(psf.kernel, image.data.shape, dtype=np.float32)
        measured = np.asarray(image.data, dtype=np.float32)
        history: List[GrayImage] = []
        for i in range(iterations):
            conv = operator.forward(estimate) + eps
            relative_blur = measured / conv
            estimate *= operator.adjoint(relative_blur)
            estimate = wiener_fft_ifft_numpy(
                estimate, psf.kernel, k, noise_psd=noise_psd
            )
            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)
            history.append(GrayImage(estimate.copy(), name=f"rl_wiener_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
        return DeconvolutionResult(history[-1], history=history, info=f"RL-Wiener iterations={iterations}; K={k}; noise_PSD={noise_psd is not None}; non_negative={non_negative}; begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, weight={tv_weight}, tv_iter={tv_iterations}")

class RichardsonLucyRosenDeconvolution(DeconvolutionAlgorithm):
    """Richardson-Lucy variant using Rosen-style nonlinear spectral correlation.

    The RL back-projection is replaced by nonlinear correlation. For two spectra F1 and F2,
    the correlation spectrum uses phase information and nonlinear magnitudes:
    |F1|^L * |F2|^M * exp(i * (phase(F1) - phase(F2))).
    L=1 and M=1 is close to ordinary correlation, while smaller exponents emphasize phase.
    """

    name = "Richardson-Lucy-Rosen"
    default_params = {
        "iterations": 20,
        "epsilon": 1e-8,
        "rosen_L": 0.5,
        "rosen_M": 0.5,
        "rosen_relax_to_one": False,
        "rosen_relax_factor": 0.98,
        "non_negative": True,
        "begin_with_wiener": False,
        "K": 0.01,
        "wiener_use_noise_psd": False,
                "use_tv_preconditioning": False,
        "tv_weight": 0.005,
        "tv_iterations": 5,
    }

    @staticmethod
    def nonlinear_correlation(a: np.ndarray, b: np.ndarray, out_shape: Tuple[int, int], L: float, M: float, eps: float) -> np.ndarray:
        padded_b = DeconvolutionAlgorithm._pad_psf(b, out_shape)
        F1 = fft2(a)
        F2 = fft2(padded_b)
        mag = (np.abs(F1) + eps) ** L * (np.abs(F2) + eps) ** M
        phase = np.exp(1j * (np.angle(F1) - np.angle(F2)))
        corr = np.real(ifft2(mag * phase))
        corr = np.nan_to_num(corr)
        corr -= corr.min()
        corr /= max(corr.max(), eps)
        return corr

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps = float(params.get("epsilon", self.default_params["epsilon"]))
        L = float(params.get("rosen_L", self.default_params["rosen_L"]))
        M = float(params.get("rosen_M", self.default_params["rosen_M"]))
        relax_to_one = bool(params.get("rosen_relax_to_one", self.default_params["rosen_relax_to_one"]))
        relax_factor = float(params.get("rosen_relax_factor", self.default_params["rosen_relax_factor"]))
        relax_factor = float(np.clip(relax_factor, 0.0, 1.0))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)

        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        k_init = float(params.get("K", self.default_params["K"]))
        estimate = wiener_fft_ifft_numpy(
            image.data, psf.kernel, k_init, noise_psd=noise_psd
        ) if begin_with_wiener else np.full_like(image.data, max(float(image.data.mean()), eps))
        if non_negative:
            estimate = np.maximum(estimate, 0.0)
        operator = NumpyLinearSameOperator(psf.kernel, image.data.shape, dtype=np.float32)
        measured = np.asarray(image.data, dtype=np.float32)
        history: List[GrayImage] = []

        current_L = L
        current_M = M
        for i in range(iterations):
            conv = operator.forward(estimate) + eps
            relative_blur = measured / conv
            rosen_backprojection = self.nonlinear_correlation(relative_blur, psf.kernel, image.data.shape, current_L, current_M, eps)
            estimate *= rosen_backprojection + eps
            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)
            history.append(GrayImage(estimate.copy(), name=f"rl_rosen_iteration_{i + 1}_L{current_L:.3f}_M{current_M:.3f}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
            if relax_to_one:
                current_L = current_L * relax_factor + (1.0 - relax_factor)
                current_M = current_M * relax_factor + (1.0 - relax_factor)

        return DeconvolutionResult(
            history[-1],
            history=history,
            info=f"RL-Rosen iterations={iterations}; L0={L}; M0={M}; relax_to_one={relax_to_one}; relax_factor={relax_factor}; final_L={current_L:.4f}; final_M={current_M:.4f}; non_negative={non_negative}; begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, weight={tv_weight}, tv_iter={tv_iterations}; neural={DeconvolutionAlgorithm._neural_denoiser_mode(params)}",
        )

class TorchBatchRichardsonLucyDeconvolution(TorchBatchedDeconvolutionMixin, DeconvolutionAlgorithm):
    name = "Torch batch Richardson-Lucy"
    default_params = {"iterations": 20, "epsilon": 1e-8, "K": 0.01, "begin_with_wiener": False, "non_negative": True,
                      "wiener_use_noise_psd": False,
                      "prefer_cuda": True, "torch_float64": False}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        params = dict(params)
        device, y, H, _, _ = self._common_batch_setup(image, psf, [params])
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps = float(params.get("epsilon", self.default_params["epsilon"]))
        x = self._initial_estimate_batch(y, H, [params], default_k=float(params.get("K", 0.01)), image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        history: List[GrayImage] = []
        for i in range(iterations):
            conv = torch.clamp(operator.forward(x), min=eps)
            ratio = y / conv
            x = x * torch.clamp(operator.adjoint(ratio), min=0.0)
            x = _torch_batch_tv_step_np_batch(x, [params])
            x = _torch_batch_neural_step_np_batch(x, [params])
            x = self._finalize_batch(x, [params])
            history.append(GrayImage(x[0].detach().cpu().numpy().astype(np.float64), name=f"torch_batch_rl_iteration_{i+1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
        neural_mode = DeconvolutionAlgorithm._neural_denoiser_mode(params)
        return DeconvolutionResult(history[-1], history=history, info=f"Torch batch RL iterations={iterations}; device={device}; convolution=linear same; neural={neural_mode}")

    def run_batch(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
        device, y, H, _, _ = self._common_batch_setup(image, psf, params_list)
        max_iter = max(int(p.get("iterations", self.default_params["iterations"])) for p in params_list)
        eps = _torch_batch_values([p.get("epsilon", self.default_params["epsilon"]) for p in params_list], self.default_params["epsilon"], "float", device)[:, None, None]
        iters = _torch_batch_values([p.get("iterations", self.default_params["iterations"]) for p in params_list], self.default_params["iterations"], "int", device)[:, None, None]
        x = self._initial_estimate_batch(y, H, params_list, default_k=0.01, image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        best = [float("-inf")] * len(params_list)
        for i in range(max_iter):
            if self._stop_requested(params_list[0]):
                break
            active = (iters > i).float()
            conv = torch.clamp(operator.forward(x), min=1e-12)
            ratio = y / torch.maximum(conv, eps)
            x_new = x * torch.clamp(operator.adjoint(ratio), min=0.0)
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

class TorchBatchRichardsonLucyWienerDeconvolution(TorchBatchRichardsonLucyDeconvolution):
    name = "Torch batch Richardson-Lucy-Wiener"
    default_params = {"iterations": 20, "epsilon": 1e-8, "K": 0.01, "begin_with_wiener": False, "non_negative": True,
                      "wiener_use_noise_psd": False,
                      "prefer_cuda": True, "torch_float64": False}

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        params = dict(params)
        device, y, H, _, _ = self._common_batch_setup(image, psf, [params])
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps = float(params.get("epsilon", self.default_params["epsilon"]))
        K = _torch_batch_values([params.get("K", self.default_params["K"])], self.default_params["K"], "float", device)
        _ignored_wiener_mode, wiener_noise = _torch_wiener_option_tensors(image, [params], y)
        x = self._initial_estimate_batch(y, H, [params], default_k=float(params.get("K", 0.01)), image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        history: List[GrayImage] = []
        for i in range(iterations):
            conv = torch.clamp(operator.forward(x), min=eps)
            ratio = y / conv
            x = x * torch.clamp(operator.adjoint(ratio), min=0.0)
            x = _torch_batch_wiener(x, H, K, noise_psd=wiener_noise)
            x = _torch_batch_tv_step_np_batch(x, [params])
            x = _torch_batch_neural_step_np_batch(x, [params])
            x = self._finalize_batch(x, [params])
            history.append(GrayImage(x[0].detach().cpu().numpy().astype(np.float64), name=f"torch_batch_rl_wiener_iteration_{i+1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
        neural_mode = DeconvolutionAlgorithm._neural_denoiser_mode(params)
        return DeconvolutionResult(history[-1], history=history, info=f"Torch batch RL-Wiener iterations={iterations}; device={device}; RL convolution=linear same; neural={neural_mode}")

    def run_batch(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
        device, y, H, _, _ = self._common_batch_setup(image, psf, params_list)
        max_iter = max(int(p.get("iterations", self.default_params["iterations"])) for p in params_list)
        eps = _torch_batch_values([p.get("epsilon", self.default_params["epsilon"]) for p in params_list], self.default_params["epsilon"], "float", device)[:, None, None]
        iters = _torch_batch_values([p.get("iterations", self.default_params["iterations"]) for p in params_list], self.default_params["iterations"], "int", device)[:, None, None]
        K = _torch_batch_values([p.get("K", self.default_params["K"]) for p in params_list], self.default_params["K"], "float", device)
        _ignored_wiener_mode, wiener_noise = _torch_wiener_option_tensors(image, params_list, y)
        x = self._initial_estimate_batch(y, H, params_list, default_k=0.01, image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        best = [float("-inf")] * len(params_list)
        for i in range(max_iter):
            if self._stop_requested(params_list[0]):
                break
            active = (iters > i).float()
            conv = torch.clamp(operator.forward(x), min=1e-12)
            ratio = y / torch.maximum(conv, eps)
            x_new = x * torch.clamp(operator.adjoint(ratio), min=0.0)
            x_new = _torch_batch_wiener(x_new, H, K, noise_psd=wiener_noise)
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

class TorchBatchRichardsonLucyRosenDeconvolution(TorchBatchRichardsonLucyDeconvolution):
    """Fast batched Torch implementation of Richardson-Lucy-Rosen for Auto.

    The forward RL blur uses the same linear-convolution ``same`` operation as
    the NumPy reference implementation.  The Rosen backprojection intentionally
    matches ``RichardsonLucyRosenDeconvolution.nonlinear_correlation``: the PSF
    is padded with the common FFT/roll convention, the nonlinear spectral
    magnitude ``|F1|^L |F2|^M`` is used, and the spatial correction is normalized
    independently for every batch item.
    """

    name = "Torch batch Richardson-Lucy-Rosen"
    default_params = {
        "iterations": 20,
        "epsilon": 1e-8,
        "K": 0.01,
        "rosen_L": 0.5,
        "rosen_M": 0.5,
        "rosen_relax_to_one": False,
        "rosen_relax_factor": 0.98,
        "begin_with_wiener": False,
        "wiener_use_noise_psd": False,
                "non_negative": True,
        "prefer_cuda": True,
        "torch_float64": False,
        "use_tv_preconditioning": False,
        "tv_weight": 0.005,
        "tv_iterations": 5,
    }

    @staticmethod
    def _rosen_backprojection_batch(relative_blur: "torch.Tensor", H: "torch.Tensor", L: "torch.Tensor", M: "torch.Tensor", eps: "torch.Tensor") -> "torch.Tensor":
        """Batched Rosen nonlinear correlation.

        relative_blur: BxHxW real tensor.
        H: BxHxW complex FFT of the padded PSF.
        L, M: B-length tensors.
        eps: Bx1x1 tensor.
        """
        F1 = torch.fft.fft2(relative_blur)
        # ``torch.angle`` and ``torch.exp(1j*...)`` are supported for complex
        # tensors; this mirrors the NumPy reference formula directly.
        Lb = L[:, None, None]
        Mb = M[:, None, None]
        mag = torch.pow(torch.abs(F1) + eps, Lb) * torch.pow(torch.abs(H) + eps, Mb)
        phase = torch.exp(1j * (torch.angle(F1) - torch.angle(H)))
        corr = torch.real(torch.fft.ifft2(mag * phase))
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        cmin = torch.amin(corr, dim=(-2, -1), keepdim=True)
        corr = corr - cmin
        cmax = torch.amax(corr, dim=(-2, -1), keepdim=True)
        return corr / torch.maximum(cmax, eps)

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        params = dict(params)
        device, y, H, _, _ = self._common_batch_setup(image, psf, [params])
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps_value = float(params.get("epsilon", self.default_params["epsilon"]))
        relax_to_one = bool(params.get("rosen_relax_to_one", self.default_params["rosen_relax_to_one"]))
        relax_factor = float(np.clip(float(params.get("rosen_relax_factor", self.default_params["rosen_relax_factor"])), 0.0, 1.0))
        x = self._initial_estimate_batch(y, H, [params], default_k=float(params.get("K", 0.01)), image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        L = torch.as_tensor([float(params.get("rosen_L", self.default_params["rosen_L"]))], dtype=x.dtype, device=device)
        M = torch.as_tensor([float(params.get("rosen_M", self.default_params["rosen_M"]))], dtype=x.dtype, device=device)
        eps = torch.as_tensor([[[eps_value]]], dtype=x.dtype, device=device)
        history: List[GrayImage] = []
        current_L = float(L.item())
        current_M = float(M.item())
        for i in range(iterations):
            conv = torch.clamp(operator.forward(x), min=eps_value)
            relative_blur = y / conv
            backprojection = self._rosen_backprojection_batch(relative_blur, H, L, M, eps)
            x = x * (backprojection + eps)
            x = _torch_batch_tv_step_np_batch(x, [params])
            x = _torch_batch_neural_step_np_batch(x, [params])
            x = self._finalize_batch(x, [params])
            history.append(GrayImage(x[0].detach().cpu().numpy().astype(np.float64), name=f"torch_batch_rl_rosen_iteration_{i+1}_L{current_L:.3f}_M{current_M:.3f}"))
            if self._iteration_completed(params, i + 1, iterations):
                break
            if relax_to_one:
                L = L * relax_factor + (1.0 - relax_factor)
                M = M * relax_factor + (1.0 - relax_factor)
                current_L = float(L.item())
                current_M = float(M.item())
        neural_mode = DeconvolutionAlgorithm._neural_denoiser_mode(params)
        return DeconvolutionResult(
            history[-1],
            history=history,
            info=f"Torch batch RL-Rosen iterations={iterations}; device={device}; convolution=linear same; L0={float(params.get('rosen_L', self.default_params['rosen_L']))}; M0={float(params.get('rosen_M', self.default_params['rosen_M']))}; relax_to_one={relax_to_one}; relax_factor={relax_factor}; final_L={current_L:.4f}; final_M={current_M:.4f}; neural={neural_mode}",
        )

    def run_batch(self, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
        device, y, H, _, _ = self._common_batch_setup(image, psf, params_list)
        max_iter = max(int(p.get("iterations", self.default_params["iterations"])) for p in params_list)
        eps = _torch_batch_values([p.get("epsilon", self.default_params["epsilon"]) for p in params_list], self.default_params["epsilon"], "float", device)[:, None, None]
        iters = _torch_batch_values([p.get("iterations", self.default_params["iterations"]) for p in params_list], self.default_params["iterations"], "int", device)[:, None, None]
        L = _torch_batch_values([p.get("rosen_L", self.default_params["rosen_L"]) for p in params_list], self.default_params["rosen_L"], "float", device)
        M = _torch_batch_values([p.get("rosen_M", self.default_params["rosen_M"]) for p in params_list], self.default_params["rosen_M"], "float", device)
        relax_to_one = _torch_batch_values([p.get("rosen_relax_to_one", self.default_params["rosen_relax_to_one"]) for p in params_list], self.default_params["rosen_relax_to_one"], "bool", device)[:, None, None]
        relax_factor = _torch_batch_values([p.get("rosen_relax_factor", self.default_params["rosen_relax_factor"]) for p in params_list], self.default_params["rosen_relax_factor"], "float", device)
        relax_factor = torch.clamp(relax_factor, 0.0, 1.0)
        x = self._initial_estimate_batch(y, H, params_list, default_k=0.01, image=image)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=x.dtype)
        best = [float("-inf")] * len(params_list)
        for i in range(max_iter):
            if self._stop_requested(params_list[0]):
                break
            active = (iters > i).float()
            conv = torch.clamp(operator.forward(x), min=1e-12)
            relative_blur = y / torch.maximum(conv, eps)
            backprojection = self._rosen_backprojection_batch(relative_blur, H, L, M, eps)
            x_new = x * (backprojection + eps)
            x_new = _torch_batch_tv_step_np_batch(x_new, params_list)
            x_new = _torch_batch_neural_step_np_batch(x_new, params_list)
            x_new = self._finalize_batch(x_new, params_list)
            x = active * x_new + (1.0 - active) * x
            if torch.any(relax_to_one > 0):
                L_relaxed = L * relax_factor + (1.0 - relax_factor)
                M_relaxed = M * relax_factor + (1.0 - relax_factor)
                mask = (relax_to_one[:, 0, 0] > 0) & (iters[:, 0, 0] > i)
                L = torch.where(mask, L_relaxed, L)
                M = torch.where(mask, M_relaxed, M)
            if reference is not None:
                scores = self._score_batch_tensor(reference, x)
                for j, s in enumerate(scores):
                    if i < int(params_list[j].get("iterations", self.default_params["iterations"])):
                        best[j] = max(best[j], s)
        if reference is None:
            best = [float("nan")] * len(params_list)
        arrays = [x[i].detach().cpu().numpy().astype(np.float64) for i in range(len(params_list))]
        return BatchedScores(scores=best, infos=arrays)

