from __future__ import annotations

from ._common import *

class TorchAdamTVMAPDeconvolution(DeconvolutionAlgorithm):
    """PyTorch/CUDA MAP deconvolution optimized directly with Adam.

    The minimized objective is:
        0.5 * ||PSF * x - y||_2^2 + tv_weight * TV(x)
    Optional non-negativity is enforced by projection after every optimizer step.
    """

    name = "PyTorch Adam TV-MAP"
    default_params = {
        "iterations": 100,
        "torch_lr": 0.05,
        "tv_weight": 0.002,
        "non_negative": True,
        "begin_with_wiener": True,
        "K": 0.01,
        "wiener_use_noise_psd": False,
                "prefer_cuda": True,
        "torch_float64": False,
        "torch_record_every": 1,
    }

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        image = self._prepare_neural_input(image, params)
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install it with CUDA support to use this algorithm.")
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        lr = float(params.get("torch_lr", self.default_params["torch_lr"]))
        tv_weight = float(params.get("tv_weight", self.default_params["tv_weight"]))
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params["begin_with_wiener"]))
        k_init = float(params.get("K", self.default_params["K"]))
        prefer_cuda = bool(params.get("prefer_cuda", self.default_params["prefer_cuda"]))
        use64 = bool(params.get("torch_float64", self.default_params.get("torch_float64", False)))
        torch_dtype = torch.float64 if use64 else torch.float32
        record_every = max(1, int(params.get("torch_record_every", self.default_params.get("torch_record_every", 1))))
        device = torch_device_name(prefer_cuda=prefer_cuda)
        if device == "unavailable":
            raise RuntimeError("PyTorch is unavailable.")

        if begin_with_wiener:
            x0_np = torch_wiener_filter_np(
                image.data, psf.kernel, k_init, device=device, torch_float64=use64,
                noise_psd=normalized_noise_psd_from_image(original_image, params),
            )
            x0_np = np.clip(x0_np, 0.0, 1.0) if non_negative else x0_np
        else:
            x0_np = image.data.copy()
        y = _torch_image(image.data, device=device, dtype=torch_dtype)
        x = _torch_image(x0_np, device=device, dtype=torch_dtype).clone().detach().requires_grad_(True)
        operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=torch_dtype)
        adam_state: Dict[str, Any] = {}
        history: List[GrayImage] = []
        last_loss = float("nan")
        for i in range(iterations):
            if x.grad is not None:
                x.grad.zero_()
            blurred = operator.forward(x)
            data_loss = 0.5 * torch.mean((blurred - y) ** 2)
            reg_loss = tv_weight * torch_tv_loss(x) if tv_weight > 0 else torch.zeros((), device=device)
            loss = data_loss + reg_loss
            loss.backward()
            with torch.no_grad():
                torch_manual_adam_step(x, adam_state, lr=lr)
                if non_negative:
                    x.clamp_(min=0.0)
                x.clamp_(max=1.5)
                if self._neural_denoiser_mode(params) == "each_iteration":
                    den = neural_denoise_np(x.detach().squeeze().cpu().numpy(), params)
                    x.copy_(_torch_image(den, device=device, dtype=torch_dtype))
            last_loss = float(loss.detach().cpu().item())
            if (i + 1) % record_every == 0 or i == iterations - 1:
                arr = x.detach().squeeze().cpu().numpy().astype(np.float64)
                history.append(GrayImage(np.clip(arr, 0.0, 1.0), name=f"torch_adam_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                if not history or history[-1].name != f"torch_adam_iteration_{i + 1}":
                    arr = x.detach().squeeze().cpu().numpy().astype(np.float64)
                    history.append(GrayImage(np.clip(arr, 0.0, 1.0), name=f"torch_adam_iteration_{i + 1}"))
                break
        result = history[-1]
        return DeconvolutionResult(
            result,
            history=history,
            info=f"PyTorch Adam TV-MAP iterations={iterations}; lr={lr}; TV weight={tv_weight}; device={device}; optimizer=manual Adam; final loss={last_loss:.6g}",
        )

def _torch_adam_tvmap_run_batch(self: TorchAdamTVMAPDeconvolution, image: GrayImage, psf: PSF, params_list: List[Dict[str, Any]], reference: Optional[GrayImage] = None, keep_history: bool = False) -> BatchedScores:
    if not params_list:
        return BatchedScores(scores=[], infos=[])
    device, torch_dtype, np_dtype = _adam_batch_device_and_dtype(params_list)
    y_np = _adam_prepare_y_stack(image, params_list, np_dtype)
    B, h, w = y_np.shape
    y = torch.as_tensor(y_np, dtype=torch_dtype, device=device)
    operator = TorchLinearSameOperator(psf.kernel, image.data.shape, device=device, dtype=torch_dtype)
    iterations = [int(p.get("iterations", self.default_params["iterations"])) for p in params_list]
    max_iter = max(iterations)
    iters_t = torch.as_tensor(iterations, dtype=torch.float32, device=device)
    lr = _torch_batch_values([p.get("torch_lr", self.default_params["torch_lr"]) for p in params_list], self.default_params["torch_lr"], "float", device).to(dtype=torch_dtype)
    tv_w = _torch_batch_values([p.get("tv_weight", self.default_params["tv_weight"]) for p in params_list], self.default_params["tv_weight"], "float", device).to(dtype=torch_dtype)
    nonneg = _torch_batch_values([p.get("non_negative", self.default_params["non_negative"]) for p in params_list], self.default_params["non_negative"], "bool", device).to(dtype=torch_dtype)[:, None, None]
    begin_w = [bool(p.get("begin_with_wiener", self.default_params["begin_with_wiener"])) for p in params_list]
    x0_list = []
    for i, p in enumerate(params_list):
        if begin_w[i]:
            k_init = float(p.get("K", self.default_params["K"]))
            x0 = torch_wiener_filter_np(
                y_np[i], psf.kernel, k_init, device=device,
                torch_float64=(torch_dtype is torch.float64),
                noise_psd=normalized_noise_psd_from_image(image, p),
            )
            if bool(p.get("non_negative", self.default_params["non_negative"])):
                x0 = np.clip(x0, 0.0, 1.0)
        else:
            x0 = y_np[i].copy()
        x0_list.append(x0.astype(np_dtype))
    x = torch.as_tensor(np.asarray(x0_list, dtype=np_dtype), dtype=torch_dtype, device=device).clone().detach().requires_grad_(True)
    state: Dict[str, Any] = {}
    history: List[GrayImage] = []
    best_scores = [float("-inf")] * B
    best_arrays = [None] * B
    record_every = max(1, int(params_list[0].get("torch_record_every", self.default_params.get("torch_record_every", 1))))
    for i in range(max_iter):
        if self._stop_requested(params_list[0]):
            break
        if x.grad is not None:
            x.grad.zero_()
        active = (iters_t > float(i)).to(dtype=torch_dtype, device=device)
        blurred = operator.forward(x)
        data_loss_per = 0.5 * torch.mean((blurred - y) ** 2, dim=(-2, -1))
        tv_per = torch_tv_loss_per_sample(x)
        loss = torch.mean(active * (data_loss_per + tv_w * tv_per))
        loss.backward()
        with torch.no_grad():
            old = x.detach().clone()
            torch_manual_adam_step_batched(x, state, lr=lr, active=active)
            x.copy_(active[:, None, None] * x + (1.0 - active[:, None, None]) * old)
            x.copy_(torch.where(nonneg > 0, torch.clamp(x, min=0.0), x))
            x.clamp_(max=1.5)
            x = _torch_batch_neural_step_np_batch(x.detach(), params_list).clone().detach().requires_grad_(True) if any(DeconvolutionAlgorithm._neural_denoiser_mode(p) == "each_iteration" for p in params_list) else x
        scores = _score_or_tv_batch(reference, x.detach(), roi_source=image, measured=y, psf=kernel)
        for j, s in enumerate(scores):
            if i < iterations[j] and s > best_scores[j]:
                best_scores[j] = float(s)
                best_arrays[j] = x[j].detach().cpu().numpy().astype(np.float64)
        if keep_history and B == 1 and ((i + 1) % record_every == 0 or i == max_iter - 1):
            history.append(GrayImage(np.clip(x[0].detach().cpu().numpy().astype(np.float64), 0.0, 1.0), name=f"torch_adam_batch_iteration_{i + 1}"))
    arrays = []
    for j in range(B):
        arr = best_arrays[j] if best_arrays[j] is not None else x[j].detach().cpu().numpy().astype(np.float64)
        arrays.append(np.clip(arr, 0.0, 1.0))
    if keep_history and B == 1 and history:
        arrays[0] = history[-1].data
    return BatchedScores(scores=best_scores, infos=arrays)

def _torch_adam_tvmap_score_batch(self: TorchAdamTVMAPDeconvolution, reference: GrayImage, degraded: GrayImage, psf: PSF, params_list: List[Dict[str, Any]]) -> List[float]:
    return self.run_batch(degraded, psf, params_list, reference=reference, keep_history=False).scores

def _torch_adam_tvmap_run(self: TorchAdamTVMAPDeconvolution, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
    # Use the batched implementation even for a single candidate. This keeps
    # Auto, Run, and future Experiment batching on the same numerical path.
    result = self.run_batch(image, psf, [dict(params)], reference=None, keep_history=True)
    arr = np.asarray(result.infos[0], dtype=np.float64)
    img = GrayImage(arr, name="torch_adam_tvmap_result")
    return DeconvolutionResult(img, history=[img], info=f"PyTorch Adam TV-MAP batched-core; iterations={int(params.get('iterations', self.default_params['iterations']))}; batch=1; device={torch_backend_device(bool(params.get('prefer_cuda', True)))}")

# Batched Auto support uses the same numerical path as the single-candidate run.
TorchAdamTVMAPDeconvolution.supports_batched_auto = True
TorchAdamTVMAPDeconvolution.run_batch = _torch_adam_tvmap_run_batch
TorchAdamTVMAPDeconvolution.score_batch = _torch_adam_tvmap_score_batch

__all__ = ["TorchAdamTVMAPDeconvolution"]
