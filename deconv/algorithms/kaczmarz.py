from __future__ import annotations

from ._common import *

class BlockKaczmarzDeconvolution(DeconvolutionAlgorithm):
    """Approximate block-Kaczmarz / ART deconvolution for convolution systems.

    Each iteration visits rectangular observation blocks. For each block, the local
    residual is back-projected with the flipped PSF and applied only to the same
    reconstruction block. This is an efficient experimental approximation of a
    row/block projection method for image deconvolution.
    """

    name = "Block Kaczmarz"
    default_params = {
        "iterations": 30,
        "kaczmarz_relaxation": 0.15,
        "kaczmarz_block_size": 32,
        "kaczmarz_blocks_per_iteration": 16,
        "kaczmarz_full_sweep": True,
        "kaczmarz_overlap": True,
        "kaczmarz_randomized": False,
        "kaczmarz_shift_grid": True,
        "kaczmarz_window": True,
        "kaczmarz_stabilized_sweep": True,
        "kaczmarz_update_damping": 0.50,
        "kaczmarz_max_update_fraction": 0.25,
        "non_negative": True,
        "begin_with_wiener": False,
        "K": 0.01,
        "wiener_use_noise_psd": False,
                "use_tv_preconditioning": False,
        "tv_weight": 0.005,
        "tv_iterations": 5,
    }

    @staticmethod
    def _starts(length: int, block_size: int, stride: int, offset: int = 0) -> List[int]:
        """Return block start positions with optional shifted grid.

        A fixed block grid can leave periodic vertical/horizontal artifacts.  The
        offset is changed between Kaczmarz epochs so the block boundaries do not
        always fall on the same pixels.  Boundary starts are still included, so
        the whole image is covered.
        """
        last = max(0, int(length) - int(block_size))
        stride = max(1, int(stride))
        offset = int(offset) % stride
        starts = [0, last]
        # Start before zero and clip to keep the same shifted lattice near edges.
        for pos in range(-offset, int(length), stride):
            starts.append(min(max(pos, 0), last))
        return sorted(set(starts))

    @staticmethod
    def _block_slices(
        height: int,
        width: int,
        block_size: int,
        overlap: bool = True,
        offset_y: int = 0,
        offset_x: int = 0,
    ) -> List[Tuple[slice, slice]]:
        block_size = max(4, int(block_size))
        stride = max(1, block_size // 2) if overlap else block_size
        blocks: List[Tuple[slice, slice]] = []
        for y0 in BlockKaczmarzDeconvolution._starts(height, block_size, stride, offset_y):
            for x0 in BlockKaczmarzDeconvolution._starts(width, block_size, stride, offset_x):
                blocks.append((slice(y0, min(y0 + block_size, height)), slice(x0, min(x0 + block_size, width))))
        return blocks

    @staticmethod
    def _window_for_block(shape: Tuple[int, int], use_window: bool) -> np.ndarray:
        """Smooth block weights to suppress seams at block boundaries."""
        bh, bw = int(shape[0]), int(shape[1])
        if (not use_window) or bh <= 2 or bw <= 2:
            return np.ones((bh, bw), dtype=np.float64)
        wy = np.hanning(bh)
        wx = np.hanning(bw)
        # Avoid zero weights at block edges; zero edges can leave uncovered pixels.
        wy = 0.25 + 0.75 * wy
        wx = 0.25 + 0.75 * wx
        return np.outer(wy, wx).astype(np.float64)

    def run(self, image: GrayImage, psf: PSF, **params: Any) -> DeconvolutionResult:
        original_image = image
        noise_psd = normalized_noise_psd_from_image(original_image, params)
        image = self._prepare_neural_input(image, params)
        iterations = int(params.get("iterations", self.default_params["iterations"]))
        relaxation = float(params.get("kaczmarz_relaxation", self.default_params["kaczmarz_relaxation"]))
        block_size = int(params.get("kaczmarz_block_size", self.default_params["kaczmarz_block_size"]))
        blocks_per_iteration = int(params.get("kaczmarz_blocks_per_iteration", self.default_params["kaczmarz_blocks_per_iteration"]))
        full_sweep = bool(params.get("kaczmarz_full_sweep", self.default_params.get("kaczmarz_full_sweep", True)))
        overlap = bool(params.get("kaczmarz_overlap", self.default_params.get("kaczmarz_overlap", True)))
        randomized = bool(params.get("kaczmarz_randomized", self.default_params["kaczmarz_randomized"]))
        shift_grid = bool(params.get("kaczmarz_shift_grid", self.default_params.get("kaczmarz_shift_grid", True)))
        use_window = bool(params.get("kaczmarz_window", self.default_params.get("kaczmarz_window", True)))
        stabilized_sweep = bool(params.get("kaczmarz_stabilized_sweep", self.default_params.get("kaczmarz_stabilized_sweep", True)))
        update_damping = float(params.get("kaczmarz_update_damping", self.default_params.get("kaczmarz_update_damping", 0.50)))
        max_update_fraction = float(params.get("kaczmarz_max_update_fraction", self.default_params.get("kaczmarz_max_update_fraction", 0.25)))
        update_damping = float(np.clip(update_damping, 0.0, 1.0))
        max_update_fraction = max(0.0, max_update_fraction)
        non_negative = bool(params.get("non_negative", self.default_params["non_negative"]))
        begin_with_wiener = bool(params.get("begin_with_wiener", self.default_params.get("begin_with_wiener", False)))
        k_init = float(params.get("K", self.default_params["K"]))
        tv_enabled, tv_weight, tv_iterations = self._tv_enabled(params)

        estimate = wiener_fft_ifft_numpy(
            image.data, psf.kernel, k_init, noise_psd=noise_psd
        ) if begin_with_wiener else image.data.copy()
        if non_negative:
            estimate = np.maximum(estimate, 0.0)

        h, w = image.data.shape
        stride = max(1, block_size // 2) if overlap else block_size
        base_blocks = self._block_slices(h, w, block_size, overlap=overlap)
        if full_sweep:
            blocks_per_iteration = len(base_blocks)
        else:
            blocks_per_iteration = max(1, min(blocks_per_iteration, len(base_blocks)))
        operator = NumpyLinearSameOperator(psf.kernel, image.data.shape, dtype=np.float32)
        psf_energy = max(float(np.sum(psf.kernel ** 2)), 1e-12)
        rng = np.random.default_rng(12345)
        history: List[GrayImage] = []

        for i in range(iterations):
            if shift_grid and full_sweep:
                # Move block boundaries between epochs.  This suppresses periodic
                # vertical/horizontal grid artifacts from a fixed ART block lattice.
                offset_y = (i * max(1, stride // 3)) % stride
                offset_x = (i * max(1, stride // 5)) % stride
                blocks = self._block_slices(h, w, block_size, overlap=overlap, offset_y=offset_y, offset_x=offset_x)
                blocks_per_iteration = len(blocks)
            else:
                blocks = base_blocks

            if randomized:
                order = rng.choice(len(blocks), size=min(blocks_per_iteration, len(blocks)), replace=False)
            else:
                if full_sweep:
                    order = list(range(len(blocks)))
                else:
                    start = (i * blocks_per_iteration) % len(blocks)
                    order = [(start + j) % len(blocks) for j in range(blocks_per_iteration)]

            # Stabilized full-sweep ART/Kaczmarz update.  Earlier versions applied
            # independent block corrections and then shifted the grid between epochs.
            # That can cause large brightness jumps and residual periodic stripes,
            # because a different block phase receives a different local normalization.
            #
            # In the stabilized mode we first build a coverage-normalized residual
            # from all selected blocks, then apply one global adjoint correction.  This
            # keeps the Kaczmarz/ART interpretation of using block observations, but
            # prevents one grid phase from dominating the whole iteration.
            blurred_estimate = operator.forward(estimate)
            residual_accumulator = np.zeros_like(image.data)
            weight_accumulator = np.zeros_like(image.data)

            for block_index in order:
                ys, xs = blocks[int(block_index)]
                residual_block = image.data[ys, xs] - blurred_estimate[ys, xs]
                window = self._window_for_block(residual_block.shape, use_window)
                residual_accumulator[ys, xs] += residual_block * window
                weight_accumulator[ys, xs] += window

            valid = weight_accumulator > 1e-12
            normalized_residual = np.zeros_like(image.data)
            normalized_residual[valid] = residual_accumulator[valid] / weight_accumulator[valid]

            if stabilized_sweep:
                correction = operator.adjoint(normalized_residual) / psf_energy
                raw_update = relaxation * correction
            else:
                # Legacy local-block mode, kept for comparison.
                update_accumulator = np.zeros_like(image.data)
                local_weight = np.zeros_like(image.data)
                for block_index in order:
                    ys, xs = blocks[int(block_index)]
                    residual_block = image.data[ys, xs] - blurred_estimate[ys, xs]
                    window = self._window_for_block(residual_block.shape, use_window)
                    residual_image = np.zeros_like(image.data)
                    residual_image[ys, xs] = residual_block * window
                    correction = operator.adjoint(residual_image) / psf_energy
                    update_accumulator[ys, xs] += correction[ys, xs] * window
                    local_weight[ys, xs] += window ** 2
                raw_update = np.zeros_like(image.data)
                local_valid = local_weight > 1e-12
                raw_update[local_valid] = relaxation * update_accumulator[local_valid] / local_weight[local_valid]

            if max_update_fraction > 0.0:
                # Limit the absolute per-iteration change relative to the current
                # dynamic range.  This avoids alternating very dark / very bright
                # epochs when the block projection is too aggressive.
                scale = max(float(np.percentile(np.abs(estimate), 99)), float(np.percentile(np.abs(image.data), 99)), 1e-6)
                raw_update = np.clip(raw_update, -max_update_fraction * scale, max_update_fraction * scale)

            proposed = estimate + raw_update
            estimate = (1.0 - update_damping) * estimate + update_damping * proposed
            if non_negative:
                estimate = np.maximum(estimate, 0.0)

            estimate = self._apply_tv_preconditioner(estimate, tv_enabled, tv_weight, tv_iterations)
            estimate = self._apply_neural_iteration_denoiser(estimate, params)
            if non_negative:
                estimate = np.maximum(estimate, 0.0)
            history.append(GrayImage(estimate.copy(), name=f"block_kaczmarz_iteration_{i + 1}"))
            if self._iteration_completed(params, i + 1, iterations):
                break

        return DeconvolutionResult(
            history[-1],
            history=history,
            info=(
                f"Block Kaczmarz iterations={iterations}; relaxation={relaxation}; "
                f"block_size={block_size}; blocks_per_iteration={blocks_per_iteration}; "
                f"full_sweep={full_sweep}; overlap={overlap}; randomized={randomized}; "
                f"shift_grid={shift_grid}; window={use_window}; stabilized={stabilized_sweep}; "
                f"damping={update_damping}; max_update_fraction={max_update_fraction}; "
                f"non_negative={non_negative}; "
                f"begin_with_wiener={begin_with_wiener}; TV={tv_enabled}, "
                f"weight={tv_weight}, tv_iter={tv_iterations}"
            ),
        )

