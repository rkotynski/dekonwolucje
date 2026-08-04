from __future__ import annotations

from ._common import *


def _rectangular_gaussian_psf(height: int, width: int, sigma: float) -> np.ndarray:
    """Return a nonnegative unit-sum Gaussian on an arbitrary rectangle."""
    height = max(1, int(height))
    width = max(1, int(width))
    sigma = max(float(sigma), 1e-6)
    yy = np.arange(height, dtype=np.float64) - (height - 1) / 2.0
    xx = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    xg, yg = np.meshgrid(xx, yy)
    kernel = np.exp(-(xg * xg + yg * yg) / (2.0 * sigma * sigma))
    return PSF.normalize_kernel(kernel)


def _blind_psf_shape(params: Dict[str, Any], image_shape: Tuple[int, int], fallback: int = 3) -> Tuple[int, int]:
    """Resolve blind-PSF height and width supplied by Tab 2."""
    height = int(params.get("blind_psf_height", 0) or 0)
    width = int(params.get("blind_psf_width", 0) or 0)
    if height <= 0 or width <= 0:
        linked = resolution_linked_psf_support(image_shape, fraction=0.45)
        height = height if height > 0 else linked
        width = width if width > 0 else linked
    height = max(1, min(int(height), int(image_shape[0])))
    width = max(1, min(int(width), int(image_shape[1])))
    return height, width

class BlindRichardsonLucyDeconvolution(DeconvolutionAlgorithm):
    """Blind Richardson-Lucy with optional physical constraints on the PSF.

    The latent image and the PSF are updated alternately.  The PSF can start
    either from a Gaussian model or from the current known/approximate PSF.  A
    rotational projection may be applied after initialization and after every
    PSF update.
    """

    name = "Blind Richardson-Lucy"
    default_params = {
        "iterations": 20,
        "epsilon": 1e-8,
        "blind_psf_height": 0,
        "blind_psf_width": 0,
        "psf_sigma": 3.0,
        "blind_psf_rotational_symmetry": False,
        "blind_use_known_psf_init": True,
        "non_negative": True,
        "begin_with_wiener": False,
        "K": 0.01,
        "wiener_use_noise_psd": False,
                "use_tv_preconditioning": False,
        "tv_weight": 0.005,
        "tv_iterations": 5,
    }

    def _initial_blind_psf(
        self,
        psf: Optional[PSF],
        image_shape: Tuple[int, int],
        psf_height: int,
        psf_width: int,
        psf_sigma: float,
        use_known: bool,
        rotational: bool,
    ) -> np.ndarray:
        """Return a centered, normalized initial PSF of the Tab-2 shape."""
        initial: Optional[np.ndarray] = None
        if use_known and psf is not None:
            try:
                fitted = calculation_psf_for_image(psf, image_shape)
                if fitted is not None:
                    initial = PSF.centered_window(
                        fitted.kernel, PSF.support_center(fitted.kernel),
                        psf_height, psf_width,
                    )
                    initial = PSF.normalize_kernel(np.maximum(initial, 0.0))
            except Exception:
                initial = None
        if initial is None:
            initial = _rectangular_gaussian_psf(psf_height, psf_width, psf_sigma)
        if rotational:
            initial = PSF.rotational_project_centered(initial)
        return PSF.normalize_kernel(np.maximum(initial, 0.0))

    def run(self, image: GrayImage, psf: Optional[PSF], **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        eps = float(params.get("epsilon", self.default_params["epsilon"]))
        psf_height, psf_width = _blind_psf_shape(params, image.data.shape)
        psf_sigma = float(params.get("psf_sigma", self.default_params["psf_sigma"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        rotational = bool(params.get(
            "blind_psf_rotational_symmetry",
            self.default_params["blind_psf_rotational_symmetry"],
        ))
        use_known = bool(params.get(
            "blind_use_known_psf_init",
            self.default_params["blind_use_known_psf_init"],
        ))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)
        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        k_init = float(params.get("K", self.default_params["K"]))

        psf_est = self._initial_blind_psf(
            psf,
            image.data.shape,
            psf_height,
            psf_width,
            psf_sigma,
            use_known=use_known,
            rotational=rotational,
        )
        estimate = wiener_fft_ifft_numpy(
            image.data, psf_est, k_init, noise_psd=noise_psd
        ) if begin_with_wiener else np.full_like(image.data, max(float(image.data.mean()), eps))
        if non_negative:
            estimate = np.maximum(estimate, 0.0)
        history: List[GrayImage] = []
        psf_history: List[np.ndarray] = []

        measured = np.asarray(image.data, dtype=np.float32)
        for i in range(iterations):
            operator = NumpyLinearSameOperator(psf_est, image.data.shape, dtype=np.float32)
            conv = operator.forward(estimate) + eps
            relative_blur = measured / conv
            estimate *= operator.adjoint(relative_blur)
            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)

            # Update PSF by correlating the current image estimate with the residual ratio.
            psf_update_full = fftconvolve(estimate[::-1, ::-1], relative_blur, mode="same")
            h, w = psf_update_full.shape
            cy, cx = h // 2, w // 2
            psf_update = PSF.centered_window(
                psf_update_full, (cy, cx), psf_height, psf_width
            )
            psf_update = np.maximum(psf_update, 0.0)
            if psf_update.shape == psf_est.shape and psf_update.sum() > eps:
                psf_est *= psf_update
                psf_est = PSF.normalize_kernel(np.maximum(psf_est, 0.0))
                if rotational:
                    psf_est = PSF.rotational_project_centered(psf_est)

            psf_history.append(psf_est.copy())
            history.append(GrayImage(
                estimate.copy(),
                name=f"blind_rl_iteration_{i + 1}",
                metadata={"estimated_psf": psf_est.copy()},
            ))
            if self._iteration_completed(params, i + 1, iterations):
                break

        if history:
            result = history[-1]
        else:
            result = GrayImage(estimate.copy(), name="blind_rl_result", metadata={"estimated_psf": psf_est.copy()})
            history = [result]
        result.metadata = dict(result.metadata or {})
        result.metadata["estimated_psf"] = psf_est.copy()
        result.metadata["estimated_psf_history"] = np.stack(psf_history, axis=0) if psf_history else psf_est[None, :, :]
        result.metadata["initial_psf"] = self._initial_blind_psf(
            psf,
            image.data.shape,
            psf_height,
            psf_width,
            psf_sigma,
            use_known=use_known,
            rotational=rotational,
        ).copy()
        init_description = "known/current PSF" if use_known and psf is not None else "Gaussian PSF"
        return DeconvolutionResult(
            result,
            history=history,
            info=(
                f"Blind RL iterations={iterations}; estimated PSF size={psf_width}x{psf_height}; "
                f"PSF size source=Tab 2; PSF initialization={init_description}; "
                f"rotational PSF={rotational}; non_negative={non_negative}; "
                f"begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, "
                f"weight={tv_weight}, tv_iter={tv_iterations}"
            ),
        )

class TorchBlindAdamTVMAPDeconvolution(DeconvolutionAlgorithm):
    """Blind PyTorch/CUDA MAP deconvolution optimized with manual Adam.

    This variant jointly estimates the latent image x and the PSF h by minimizing
        0.5 * ||h * x - y||_2^2 + lambda_x TV(x) + lambda_h TV(h)
    with projection of h after every iteration to a nonnegative, sum-one PSF.
    Optionally the PSF projection includes rotational/radial symmetry.
    """

    name = "PyTorch Blind Adam TV-MAP"
    default_params = {
        "iterations": 150,
        "torch_lr": 0.03,
        "blind_psf_lr": 0.01,
        "tv_weight": 0.002,
        "blind_psf_tv_weight": 0.0005,
        "blind_psf_height": 0,
        "blind_psf_width": 0,
        "psf_sigma": 4.0,
        "blind_psf_rotational_symmetry": False,
        "blind_use_known_psf_init": True,
        "non_negative": True,
        "begin_with_wiener": False,
        "K": 0.01,
        "wiener_use_noise_psd": False,
                "prefer_cuda": True,
        "torch_float64": False,
        "torch_record_every": 1,
    }

    def _initial_blind_psf(self, psf: Optional[PSF], image_shape: Tuple[int, int], params: Dict[str, Any]) -> np.ndarray:
        height, width = _blind_psf_shape(params, image_shape)
        use_known = bool(params.get(
            "blind_use_known_psf_init",
            self.default_params["blind_use_known_psf_init"],
        ))
        if use_known and psf is not None:
            try:
                fitted = calculation_psf_for_image(psf, image_shape)
                if fitted is not None:
                    k = PSF.centered_window(
                        fitted.kernel, PSF.support_center(fitted.kernel), height, width
                    )
                    return PSF.normalize_kernel(np.maximum(k, 0.0))
            except Exception:
                pass
        sigma = float(params.get("psf_sigma", self.default_params["psf_sigma"]))
        return _rectangular_gaussian_psf(height, width, sigma)

    def run(self, image: GrayImage, psf: Optional[PSF], **params: Any) -> DeconvolutionResult:
        original_image = image
        image = self._prepare_neural_input(image, params)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install it with CUDA support to use this algorithm.")
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        lr_x = float(params.get("torch_lr", self.default_params["torch_lr"]))
        lr_h = float(params.get("blind_psf_lr", self.default_params["blind_psf_lr"]))
        tv_weight = float(params.get("tv_weight", self.default_params["tv_weight"]))
        psf_tv_weight = float(params.get("blind_psf_tv_weight", self.default_params["blind_psf_tv_weight"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params["begin_with_wiener"]))
        k_init = float(params.get("K", self.default_params["K"]))
        prefer_cuda = bool(params.get("prefer_cuda", self.default_params["prefer_cuda"]))
        use64 = bool(params.get("torch_float64", self.default_params.get("torch_float64", False)))
        torch_dtype = torch.float64 if use64 else torch.float32
        np_dtype = np.float64 if use64 else np.float32
        rotational = bool(params.get("blind_psf_rotational_symmetry", self.default_params["blind_psf_rotational_symmetry"]))
        use_known = bool(params.get("blind_use_known_psf_init", self.default_params["blind_use_known_psf_init"]))
        record_every = max(1, int(params.get("torch_record_every", self.default_params.get("torch_record_every", 1))))
        device = torch_device_name(prefer_cuda=prefer_cuda)
        if device == "unavailable":
            raise RuntimeError("PyTorch is unavailable.")

        h0_np = self._initial_blind_psf(psf, image.data.shape, params)
        if rotational:
            h0_np = PSF.rotational_project_centered(h0_np)
        if begin_with_wiener:
            x0_np = torch_wiener_filter_np(
                image.data, h0_np, k_init, device=device, torch_float64=use64,
                noise_psd=normalized_noise_psd_from_image(original_image, params),
            )
            x0_np = np.clip(x0_np, 0.0, 1.0) if non_negative else x0_np
        else:
            x0_np = image.data.copy()

        y = _torch_image(image.data, device=device, dtype=torch_dtype)
        x = _torch_image(x0_np, device=device, dtype=torch_dtype).clone().detach().requires_grad_(True)
        h = torch.as_tensor(np.asarray(h0_np, dtype=np_dtype), dtype=torch_dtype, device=device).clone().detach().requires_grad_(True)
        torch_project_psf_(h, rotational_symmetry=rotational)
        x_state: Dict[str, Any] = {}
        h_state: Dict[str, Any] = {}
        history: List[GrayImage] = []
        psf_history: List[PSF] = []
        last_loss = float("nan")

        for i in range(iterations):
            if x.grad is not None:
                x.grad.zero_()
            if h.grad is not None:
                h.grad.zero_()
            blurred = torch_conv_same_tensor(x, h, flip=True)
            data_loss = 0.5 * torch.mean((blurred - y) ** 2)
            reg_x = tv_weight * torch_tv_loss(x) if tv_weight > 0 else torch.zeros((), device=device)
            reg_h = psf_tv_weight * torch_tv_loss(h[None, None, :, :]) if psf_tv_weight > 0 else torch.zeros((), device=device)
            loss = data_loss + reg_x + reg_h
            loss.backward()
            with torch.no_grad():
                torch_manual_adam_step(x, x_state, lr=lr_x)
                torch_manual_adam_step(h, h_state, lr=lr_h)
                if non_negative:
                    x.clamp_(min=0.0)
                x.clamp_(max=1.5)
                torch_project_psf_(h, rotational_symmetry=rotational)
                if self._neural_denoiser_mode(params) == "each_iteration":
                    den = neural_denoise_np(x.detach().squeeze().cpu().numpy(), params)
                    x.copy_(_torch_image(den, device=device, dtype=torch_dtype))
            last_loss = float(loss.detach().cpu().item())
            if (i + 1) % record_every == 0 or i == iterations - 1:
                arr = x.detach().squeeze().cpu().numpy().astype(np.float64)
                hk = h.detach().cpu().numpy().astype(np.float64)
                hk = PSF.normalize_kernel(np.maximum(hk, 0.0))
                history.append(GrayImage(
                    np.clip(arr, 0.0, 1.0),
                    name=f"torch_blind_adam_iteration_{i + 1}",
                    metadata={"estimated_psf": hk.copy()},
                ))
                psf_history.append(PSF(hk.copy(), name=f"estimated_psf_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                # Preserve the last completed state even when it did not coincide
                # with the configured recording interval.
                if not history or history[-1].name != f"torch_blind_adam_iteration_{i + 1}":
                    arr = x.detach().squeeze().cpu().numpy().astype(np.float64)
                    hk = PSF.normalize_kernel(np.maximum(h.detach().cpu().numpy().astype(np.float64), 0.0))
                    history.append(GrayImage(
                        np.clip(arr, 0.0, 1.0),
                        name=f"torch_blind_adam_iteration_{i + 1}",
                        metadata={"estimated_psf": hk.copy()},
                    ))
                    psf_history.append(PSF(hk.copy(), name=f"estimated_psf_iteration_{i + 1}"))
                break

        estimated_psf = psf_history[-1] if psf_history else PSF(h0_np, name="estimated_psf")
        result = history[-1] if history else GrayImage(np.clip(x0_np, 0.0, 1.0), name="torch_blind_adam_result")
        result.metadata = dict(result.metadata or {})
        result.metadata["estimated_psf"] = estimated_psf.kernel.copy()
        result.metadata["estimated_psf_history"] = np.stack([p.kernel for p in psf_history], axis=0) if psf_history else np.empty((0, 0, 0))
        result.metadata["initial_psf"] = h0_np.copy()
        return DeconvolutionResult(
            result,
            history=history,
            info=(
                f"PyTorch Blind Adam TV-MAP iterations={iterations}; lr_x={lr_x}; lr_psf={lr_h}; "
                f"TV image={tv_weight}; TV PSF={psf_tv_weight}; rotational PSF={rotational}; "
                f"PSF initialization={'known/current PSF' if use_known and psf is not None else 'Gaussian PSF'}; "
                f"device={device}; optimizer=manual Adam; final loss={last_loss:.6g}; "
                f"estimated {describe_psf_kernel(estimated_psf, 'estimated PSF')}"
            ),
        )

def _torch_blind_adam_tvmap_run_batch(self: TorchBlindAdamTVMAPDeconvolution, image: GrayImage, psf: Optional[PSF], params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
    if not params_list:
        return BatchedScores(scores=[], infos=[])
    device, torch_dtype, np_dtype = _adam_batch_device_and_dtype(params_list)
    y_np = _adam_prepare_y_stack(image, params_list, np_dtype)
    B, h_img, w_img = y_np.shape
    y = torch.as_tensor(y_np, dtype=torch_dtype, device=device)
    # Batch blind PSF requires common tensor shape. Use the largest requested support in the chunk.
    h0s = []
    shapes = []
    rotational_flags = []
    for p in params_list:
        h0 = self._initial_blind_psf(psf, image.data.shape, p)
        if bool(p.get("blind_psf_rotational_symmetry", self.default_params["blind_psf_rotational_symmetry"])):
            h0 = PSF.rotational_project_centered(h0)
        h0s.append(h0)
        shapes.append(tuple(int(v) for v in h0.shape))
        rotational_flags.append(bool(p.get("blind_psf_rotational_symmetry", self.default_params["blind_psf_rotational_symmetry"])))
    common_h = max(shape[0] for shape in shapes)
    common_w = max(shape[1] for shape in shapes)
    h_stack = []
    for h0 in h0s:
        h_stack.append(PSF.normalize_kernel(np.maximum(
            PSF.centered_window(h0, PSF.support_center(h0), common_h, common_w), 0.0
        )).astype(np_dtype))
    h_var = torch.as_tensor(np.asarray(h_stack, dtype=np_dtype), dtype=torch_dtype, device=device).clone().detach().requires_grad_(True)
    torch_project_psf_batch_(h_var, rotational_flags)
    iterations = [int(p.get("iterations", self.default_params["iterations"])) for p in params_list]
    max_iter = max(iterations)
    iters_t = torch.as_tensor(iterations, dtype=torch.float32, device=device)
    lr_x = _torch_batch_values([p.get("torch_lr", self.default_params["torch_lr"]) for p in params_list], self.default_params["torch_lr"], "float", device).to(dtype=torch_dtype)
    lr_h = _torch_batch_values([p.get("blind_psf_lr", self.default_params["blind_psf_lr"]) for p in params_list], self.default_params["blind_psf_lr"], "float", device).to(dtype=torch_dtype)
    tv_w = _torch_batch_values([p.get("tv_weight", self.default_params["tv_weight"]) for p in params_list], self.default_params["tv_weight"], "float", device).to(dtype=torch_dtype)
    psf_tv_w = _torch_batch_values([p.get("blind_psf_tv_weight", self.default_params["blind_psf_tv_weight"]) for p in params_list], self.default_params["blind_psf_tv_weight"], "float", device).to(dtype=torch_dtype)
    nonneg = _torch_batch_values([p.get("non_negative", self.default_params["non_negative"]) for p in params_list], self.default_params["non_negative"], "bool", device).to(dtype=torch_dtype)[:, None, None]
    x0_list = []
    for i, p in enumerate(params_list):
        if bool(p.get("begin_with_wiener", self.default_params["begin_with_wiener"])):
            x0 = torch_wiener_filter_np(
                y_np[i], h_stack[i], float(p.get("K", self.default_params["K"])),
                device=device, torch_float64=(torch_dtype is torch.float64),
                noise_psd=normalized_noise_psd_from_image(image, p),
            )
            if bool(p.get("non_negative", self.default_params["non_negative"])):
                x0 = np.clip(x0, 0.0, 1.0)
        else:
            x0 = y_np[i].copy()
        x0_list.append(x0.astype(np_dtype))
    x = torch.as_tensor(np.asarray(x0_list, dtype=np_dtype), dtype=torch_dtype, device=device).clone().detach().requires_grad_(True)
    x_state: Dict[str, Any] = {}
    h_state: Dict[str, Any] = {}
    best_scores = [float("-inf")] * B
    best_arrays = [None] * B
    best_psfs = [None] * B
    for i in range(max_iter):
        if self._stop_requested(params_list[0]):
            break
        if x.grad is not None:
            x.grad.zero_()
        if h_var.grad is not None:
            h_var.grad.zero_()
        active = (iters_t > float(i)).to(dtype=torch_dtype, device=device)
        blurred_list = []
        for j in range(B):
            blurred_list.append(_torch_fftconvolve_same_batch(x[j:j+1], h_var[j])[0])
        blurred = torch.stack(blurred_list, dim=0)
        data_loss_per = 0.5 * torch.mean((blurred - y) ** 2, dim=(-2, -1))
        tv_per = torch_tv_loss_per_sample(x)
        psf_tv_per = torch_tv_loss_per_sample(h_var)
        loss = torch.mean(active * (data_loss_per + tv_w * tv_per + psf_tv_w * psf_tv_per))
        loss.backward()
        with torch.no_grad():
            old_x = x.detach().clone()
            old_h = h_var.detach().clone()
            torch_manual_adam_step_batched(x, x_state, lr=lr_x, active=active)
            torch_manual_adam_step_batched(h_var, h_state, lr=lr_h, active=active)
            x.copy_(active[:, None, None] * x + (1.0 - active[:, None, None]) * old_x)
            h_var.copy_(active[:, None, None] * h_var + (1.0 - active[:, None, None]) * old_h)
            x.copy_(torch.where(nonneg > 0, torch.clamp(x, min=0.0), x))
            x.clamp_(max=1.5)
            torch_project_psf_batch_(h_var, rotational_flags)
            x = _torch_batch_neural_step_np_batch(x.detach(), params_list).clone().detach().requires_grad_(True) if any(DeconvolutionAlgorithm._neural_denoiser_mode(p) == "each_iteration" for p in params_list) else x
            h_var = h_var.detach().clone().requires_grad_(True)
        scores = _score_or_tv_batch(reference, x.detach(), roi_source=image, measured=y, psf=h_var.detach())
        for j, s in enumerate(scores):
            if i < iterations[j] and s > best_scores[j]:
                best_scores[j] = float(s)
                best_arrays[j] = x[j].detach().cpu().numpy().astype(np.float64)
                best_psfs[j] = h_var[j].detach().cpu().numpy().astype(np.float64)
    arrays = []
    for j in range(B):
        arr = best_arrays[j] if best_arrays[j] is not None else x[j].detach().cpu().numpy().astype(np.float64)
        hk = best_psfs[j] if best_psfs[j] is not None else h_var[j].detach().cpu().numpy().astype(np.float64)
        hk = PSF.normalize_kernel(np.maximum(hk, 0.0))
        arrays.append({
            "image": np.clip(arr, 0.0, 1.0),
            "estimated_psf": hk,
            "initial_psf": np.asarray(h_stack[j], dtype=np.float64).copy(),
        })
    return BatchedScores(scores=best_scores, infos=arrays)

def _torch_blind_adam_tvmap_score_batch(self: TorchBlindAdamTVMAPDeconvolution, reference: GrayImage, degraded: GrayImage, psf: Optional[PSF], params_list: List[Dict[str, Any]]) -> List[float]:
    return self.run_batch(degraded, psf, params_list, reference=reference, keep_history=False).scores

def _torch_blind_adam_tvmap_run(self: TorchBlindAdamTVMAPDeconvolution, image: GrayImage, psf: Optional[PSF], **params: Any) -> DeconvolutionResult:
    result = self.run_batch(image, psf, [dict(params)], reference=None, keep_history=True)
    info0 = result.infos[0]
    arr = np.asarray(info0["image"], dtype=np.float64) if isinstance(info0, dict) else np.asarray(info0, dtype=np.float64)
    hk = np.asarray(info0.get("estimated_psf"), dtype=np.float64) if isinstance(info0, dict) else None
    img = GrayImage(arr, name="torch_blind_adam_tvmap_result")
    if hk is not None:
        img.metadata = {
            "estimated_psf": hk.copy(),
            "estimated_psf_history": hk[None, :, :].copy(),
        }
        if isinstance(info0, dict) and info0.get("initial_psf") is not None:
            img.metadata["initial_psf"] = np.asarray(info0["initial_psf"], dtype=np.float64).copy()
    init_mode = "known/current PSF" if bool(params.get("blind_use_known_psf_init", self.default_params["blind_use_known_psf_init"])) and psf is not None else "Gaussian PSF"
    rotational = bool(params.get("blind_psf_rotational_symmetry", self.default_params["blind_psf_rotational_symmetry"]))
    return DeconvolutionResult(img, history=[img], info=f"PyTorch Blind Adam TV-MAP batched-core; iterations={int(params.get('iterations', self.default_params['iterations']))}; batch=1; device={torch_backend_device(bool(params.get('prefer_cuda', True)))}; PSF initialization={init_mode}; rotational PSF={rotational}")

# Batched Auto support for the blind Adam estimator.
TorchBlindAdamTVMAPDeconvolution.supports_batched_auto = True
TorchBlindAdamTVMAPDeconvolution.run_batch = _torch_blind_adam_tvmap_run_batch
TorchBlindAdamTVMAPDeconvolution.score_batch = _torch_blind_adam_tvmap_score_batch

__all__ = ["BlindRichardsonLucyDeconvolution", "TorchBlindAdamTVMAPDeconvolution"]
