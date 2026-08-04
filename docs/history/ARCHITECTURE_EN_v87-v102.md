# Project architecture

## Numerical core

`deconv/core/operators.py` contains the shared convolution/FFT layer. It
separates zero-boundary linear `same` convolution from circular FFT convolution,
provides exact adjoints, and caches fixed-PSF spectra for NumPy and Torch/CUDA.
Torch operators use float32 by default.

`deconv/core/runtime.py` contains data structures and the remaining reusable
numerical operations shared by several algorithms:

- `GrayImage` and `PSF`;
- metric calculation;
- PSF fitting and normalization;
- NumPy/SciPy and PyTorch convolution helpers;
- TV and denoiser integration helpers;
- base result and algorithm interfaces.

It does not import PyQt5 and can be used in scripts or tests without starting
the GUI.

## Algorithms

Concrete algorithms are implemented in `deconv/algorithms/`:

- `wiener.py` — Wiener and batched Torch Wiener;
- `richardson_lucy.py` — Richardson–Lucy, RL–Wiener, RL–Rosen and Torch batch variants;
- `landweber.py` — Landweber, Wiener-preconditioned Landweber and Torch batch Landweber;
- `blind.py` — blind Richardson–Lucy and blind Adam TV-MAP;
- `adam.py` — Adam TV-MAP;
- `kaczmarz.py` — stabilized block Kaczmarz/ART;
- `registry.py` — construction and lookup of algorithms;
- `base.py` — public algorithm interface types.

No file in `deconv/algorithms/` imports `legacy_runtime.py`.

## GUI and compatibility layer

`deconv/legacy_runtime.py` now contains the Qt widgets, application state and
backward-compatible entry point.  It imports algorithms from
`deconv.algorithms` rather than defining them.

The name is retained for compatibility with existing imports and saved user
workflows.  A future refactoring stage can split its GUI classes into
`deconv/gui/` without changing the numerical modules.

## Adding an algorithm

1. Add a class derived from `DeconvolutionAlgorithm` in an appropriate file in
   `deconv/algorithms/`.
2. Implement `run()` and optionally `run_batch()` / `score_batch()`.
3. Register the class in `deconv/algorithms/registry.py`.
4. Add its parameter controls and visibility mapping in the GUI.

## Metric region

`GrayImage.from_array_with_zero_frame()` stores a `content_roi` tuple in image
metadata. `compute_metrics()` and the Torch batch scoring utilities use this
rectangle for PSNR, SSIM and TV. GUI code passes the measured/degraded image as
the ROI source when no independent reference exists.

## Blind PSF initialization and projection

`deconv/algorithms/blind.py` owns the shared blind-PSF behavior. Both blind
algorithms accept `blind_use_known_psf_init` and
`blind_psf_rotational_symmetry`. The known PSF is treated only as an initial
estimate; it remains an optimized variable. Rotational symmetry is a projection
applied after each PSF update, not merely a display operation.

## Numerical worker coordination (v84)

Qt worker threads do not access GUI controls while tuning. Auto receives a
snapshot of the selected parameter values and uses plain Python limits. Auto and
Test runs acquire a shared numerical-work coordinator before entering algorithm
code. CUDA synchronization and allocator cleanup are performed in the same
worker thread before it exits. This prevents concurrent CUDA use and avoids
releasing QThread/QObject wrappers while native kernels are still active.

## Display levels and threshold snapshots (v92)

Test-tab display clipping is isolated from numerical reconstruction.  The black
and white sliders select direct intensity levels.  A cached 4096-bin histogram
and CDF provide approximate percentile labels without repeated sorting.  Slider
movement updates only the existing Matplotlib image artist; metrics and PSF
views are cached or left unchanged.  The underlying `GrayImage`, saved result,
metrics and Auto scores remain unchanged.

Threshold previews retain immutable snapshots of the measured image, current
PSF and degradation PSF. Reset restores those three records exactly and then
clears the snapshots, preventing cumulative thresholding or stale paired-PSF
state across preview sessions.

## Wiener-specific no-reference model selection (v86)

Plain Wiener Auto is special-cased before reconstruction scoring. Because the
Wiener fitted-data operator is diagonal in the FFT basis, `wiener_gcv_cost()`
computes generalized cross-validation from the measured spectrum, PSF transfer
function and optional normalized noise PSD. This path avoids constructing and
normalizing a candidate `GrayImage`, which previously made the data-consistency
term unsuitable for selecting K.

Manual K scanning is coordinated by `DeconvolutionRunWorker`. For Wiener and
Torch batch Wiener it generates logarithmically spaced K values and runs one
complete reconstruction per value. Each returned frame stores `wiener_K`,
`wiener_gcv`, scan index and scan length in metadata. The Test tab uses this
metadata for browsing, best-frame selection, logging and copying a selected K
back to the Algorithm tab.

The generic no-reference metric now includes `RESIDUAL_WHITENESS`, calculated
from normalized residual autocorrelation at six short spatial lags. CPU and
Torch batch score paths use matching weights.


## Convolution models and PSF provenance (v87)

Synthetic degradation and non-blind iterative reconstruction use
`NumpyLinearSameOperator` or `TorchLinearSameOperator`. Their forward operation
matches `scipy.signal.fftconvolve(..., mode="same")` with zero boundary
conditions; their adjoint is the transpose of that exact cropped operation.

Closed-form Wiener filtering uses `circular_fft`, because its transfer function
is diagonal only for circular convolution on the selected image grid. PSF
metadata therefore distinguishes `convolution_model` (the physical/reblur
model) from `algorithm_convolution_model` (the model actually inverted by the
algorithm). The reported `linear_vs_circular_input_mismatch` quantifies the
boundary-model discrepancy for the current measured image and kernel.

`calculation_psf_for_image()` is the only PSF preparation path used by numerical algorithms. It reads the thresholded full PSF and the rectangular centre/size metadata from Tab 2, extracts that window, pads missing samples with zeros, projects to nonnegative values and normalizes the compact kernel to unit sum. `state["calculation_psf"]` is the object shown in Tabs 1 and 2 and passed to reconstruction, automatic tuning, residual metrics and synthetic degradation.

Stored forward/degradation PSFs may remain only as diagnostic records. They are never selectable as reconstruction inputs. The legacy `reconstruction_psf_for_image()` signature is retained for third-party callers, but ignores its old snapshot arguments and delegates to `calculation_psf_for_image()`.

## Shared Wiener profile (v99)

The plain Wiener profile supplies only parameters common to Wiener stages: `K`, optional normalized noise PSD and real-versus-absolute IFFT output. Every Wiener stage uses the same current calculation PSF. NumPy execution calls the explicit `wiener_fft_ifft_numpy()` formula, while Torch execution uses `torch.fft.fft2` and `torch.fft.ifft2`; no dedicated restoration routine is used.

Blind Richardson–Lucy and blind Adam receive `blind_psf_width` and `blind_psf_height` from the Tab-2 rectangle on every run, including automatic tuning. The known calculation PSF may initialize the estimate, but does not change its selected array dimensions.

## Test-tab iteration and display-range controls (v89)

The Test tab keeps all recorded `GrayImage` frames in `TestTab.history`. The
**Select best iteration** action therefore reruns only the selection criterion,
not the deconvolution. It uses the same `best_iteration_index()` path as the
automatic post-run selection.

Percentile optimization is separated into a generic numerical search helper,
`optimize_percentile_range()`, and GUI-specific scoring. The GUI converts the
current raw reconstruction to the exact image that would be displayed for a
candidate slider pair, then scores that clipped/rescaled image. Reference data
use PSNR; measured data use the existing no-reference reblur/TV/whiteness cost.
The raw reconstruction stored in the history is never modified. Common-scale
mode derives candidate black/white levels from all history frames before scoring
the currently selected frame.



## PSF selection overlay and batched history metrics (v93)

`ImageCanvas.show_image()` accepts an optional rectangle in image coordinates.
Tab 2 derives that rectangle from the same center and odd support width used by
`PSF.centered_window()`, so the preview and the numerical crop cannot drift apart.
The rectangle may extend beyond the loaded PSF array because the calculation
window is zero-padded in exactly that situation.

`compute_metrics_batch()` evaluates a complete Test-tab history with a leading
iteration dimension. Torch float32 computes TV/NTV, PSNR, reblur residual,
intensity error and residual whiteness. Linear and circular FFT reblurring group
frames by PSF shape and convolution model; kernels may still differ between
frames, which supports blind-PSF histories. SSIM uses SciPy uniform filters with
a batch-preserving filter size. CUDA is preferred when requested, with an
adaptive memory budget and automatic CPU or scalar fallback. TestTab preloads
its metric cache from this batch before selecting the best frame.


## Interactive PSF selection and numeric editors (v94)

The full-PSF `ImageCanvas` owns a lightweight mouse interaction layer for the
calculation rectangle. A press inside the rectangle starts translation, while a
press within nine display pixels of its border starts symmetric square resize.
The released integer geometry is returned to `DegradedInputTab`, which switches
the selection to manual mode and updates the centre and odd support width. The
actual convolution path remains authoritative: Apply writes
`calculation_center_mode="manual"` and `calculation_center=(y, x)` into PSF
metadata, and `PSF.fitted_to_shape()` performs the same zero-padded crop.

The GUI aliases Qt spin boxes with small subclasses. Keyboard tracking is
disabled so intermediate text is not sent to refresh slots, double fields use
15 internal decimal places with compact trailing-zero-free rendering, and both
integer and floating editors have wider text areas. This changes editing
behaviour without changing the parameter dictionaries consumed by algorithms.

After `compute_metrics_batch()` fills the history metric cache, TestTab selects
the best frame and invokes cached-CDF Auto levels. Display-level selection is
therefore automatic but remains independent of the reconstruction criterion.


## Persistent image directory and image extent correction (v95)

`LoadGenerateTab.settings()` stores `last_image_directory` in the active JSON
profile. Every image-like open dialog starts from that directory and updates it
after a file is selected. Missing or obsolete paths fall back to the user's home
directory.

`AxesImage.set_data()` changes the array but deliberately preserves the previous
image extent. Since the PSF preview switches between arrays of different sizes,
that behaviour used to leave the full PSF occupying the old crop-sized rectangle
in the upper-left part of the axes. `ImageCanvas.show_image()` now resets the
extent to `(-0.5, width-0.5, height-0.5, -0.5)` for every update. The image, axes
limits and editable selection rectangle therefore share identical pixel
coordinates.


## Known-PSF selection and joint floor/K tuning (v96)

Tab 1 owns only the full loaded/generated PSF array and generation parameters.
`PSF.automatic_support_selection()` estimates a conservative almost-nonzero
window from robust perimeter background statistics. Tab 2 stores the selected
centre and odd square width in PSF metadata and application state.
`PSF.fitted_to_shape()` extracts exactly that window, zero-pads it when needed
and normalizes the resulting kernel to unit sum.

`optimize_psf_floor_and_wiener_k()` jointly scans the floor/peak ratio and
Wiener K on a reduced preview. It minimizes reference MSE when an independent
reference exists and Wiener GCV otherwise. The GUI saves the selected K in the
plain-Wiener profile; propagation to auxiliary Wiener stages remains an explicit
user action.


## Rectangular known-PSF selection and joint tuning (v97)

The Tab-2 selection is represented by `calculation_support_height`, `calculation_support_width`, `calculation_center_mode`, and `calculation_center`. `PSF.fitted_to_shape()` extracts that rectangle with zero padding where necessary and normalizes it. The scalar support signal is only a conservative maximum extent for legacy padding policies and must not overwrite the rectangular state.

`_prepare_psf_candidate_window()` is the authoritative candidate-preparation path for the joint floor/K search: threshold, rectangular crop, then unit-sum normalization. Reference-based tuning minimizes Wiener reconstruction MSE; no-reference tuning uses full-spectrum Wiener GCV.


## Auto cancellation isolation (v101)

The Qt `AutoTuneWorker` only orchestrates candidate generation. Numerical scoring is delegated to one persistent `multiprocessing` spawn process through `deconv.optim.auto_process.AutoNumericalProcessClient`. A shared event provides cooperative stopping at iteration boundaries. The client polls the process connection and terminates only the helper process when five seconds have elapsed since cancellation. This avoids asynchronous termination of a Python `QThread`, which could otherwise leave `_NUMERICAL_WORK_LOCK` or a CUDA context in an undefined state.


## Frozen Auto feature state and constrained PSF-floor tuning (v102)

`auto_tunable_parameter_names()` derives the tunable parameter set once from the initial algorithm profile. Feature activation controls are frozen, and dependent values are excluded when Wiener initialization, the denoiser, optional TV processing, or Rosen relaxation is inactive. The frozen list is carried as an internal Auto parameter and gates every scalar, quadratic, and full-batch candidate generator.

No-reference PSF-floor tuning no longer compares unconstrained GCV across arbitrary transfer functions. `_psf_floor_background_statistics()` estimates a robust admissible interval from the selected PSF-frame perimeter. For each admissible floor, `_prepare_psf_candidate_window()` performs the same threshold/crop/unit-normalization sequence as Tab 2, and GCV selects K conditionally on that fixed kernel. A PSF-background prior and support-collapse diagnostics prevent nearly impulsive high-floor solutions. Reference-backed tuning continues to minimize reconstruction MSE directly.
