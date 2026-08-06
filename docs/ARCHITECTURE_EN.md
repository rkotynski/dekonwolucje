# Architecture and developer guide

## Repository layout

- `deconv/core/` — GUI-independent image/PSF models, FFT and convolution operators, metrics, thresholds, display-level helpers and common numerical utilities.
- `deconv/algorithms/` — concrete deconvolution implementations and registry.
- `deconv/optim/` — GUI-independent Auto tuning, batch helpers, and the isolated GUI numerical worker process.
- `deconv/denoisers/` — optional neural-denoiser architectures and loading helpers.
- `deconv/gui/` — GUI adapters, including translated Qt widgets.
- `deconv/api.py` — stable Qt-independent entry points for scripts, notebooks, and batch jobs.
- `examples/` — standalone programs using the public API.
- `deconv/legacy_runtime.py` — stabilized Qt application state and tab implementation retained for backward compatibility.
- `tests/` — numerical and process-level regression tests.
- `docs/` — bilingual documentation.

## Internationalization

The application supports exactly `en` and `pl`. English source strings are stable translation identifiers in `deconv/i18n.py`. `deconv/gui/translated_widgets.py` stores the English source text for widgets, combo-box entries, form labels, dialogs and logs, then retranslates them in place.

Internal values remain independent of language:

- algorithm names stored in settings are English registry identifiers;
- combo boxes display translated text but return canonical English source values to application logic;
- configuration keys and source-code comments remain English;
- `language` is the only language-dependent profile field.

To correct a translation, edit `_EXACT_PL` or the generated-message phrase table in `deconv/i18n.py`. No `.ts`/`.qm` compilation step is required because only two languages are planned.

## Numerical data flow

The state contains source arrays used for Reset and a single committed calculation pair:

- the thresholded input image;
- `calculation_psf`, prepared by thresholding, rectangular cropping, nonnegative projection and unit-sum normalization.

All algorithms, Wiener initialization, blind-PSF initialization, reblur metrics and synthetic degradation use this explicit calculation PSF.

## Convolution models

`deconv/core/operators.py` distinguishes:

- `linear_same`: zero-boundary linear convolution with an exact adjoint;
- `circular_fft`: circular convolution on the image grid, used by the explicit FFT/IFFT Wiener inverse.

Fixed PSF spectra are cached. Torch operators use `float32` by default.

## Block Kaczmarz implementation notes

`deconv/algorithms/kaczmarz.py` implements an approximate block ART method on top of `NumpyLinearSameOperator`. The forward convolution is evaluated once per outer iteration. Selected observation-block residuals are accumulated with optional raised-Hann windows and normalized by block coverage. In the stabilized mode, one global adjoint correction is divided by the PSF energy. The legacy local mode applies block-local adjoint corrections and normalizes overlapping contributions. Shifted block starts, deterministic or random ordering, update clipping, damping, nonnegativity, TV, and denoising are separate controls. The implementation deliberately avoids materializing the convolution matrix and should not be described as an exact solution of block normal equations.


## Public numerical API

`deconv.api` is the supported integration layer outside the GUI. It converts NumPy arrays to `GrayImage`/`PSF`, exposes the canonical algorithm registry, merges default and user parameters, prepares the calculation-safe PSF, and returns the native `DeconvolutionResult`. It imports no Qt modules.

The stable functions include `run_deconvolution`, `auto_tune_parameters`, `auto_deconvolve`, the algorithm convenience wrappers, the PSF generators, `disturb_image`, conversion helpers, and `save_grayscale`. `deconv/optim/auto_api.py` contains the Qt-independent Auto engine; GUI state and `legacy_runtime.py` are not public numerical APIs.

See `docs/API_EN.md`, `examples/wiener_motion_blur.py`, and `examples/auto_richardson_lucy_motion.py`.

## Adding an algorithm

1. Derive from `DeconvolutionAlgorithm` in `deconv/algorithms/`.
2. Implement `run()` and optionally batched scoring/run methods.
3. Register the class in `deconv/algorithms/registry.py`.
4. Add canonical English parameter controls in the GUI.
5. Add Polish translations for every new visible source string.
6. Add numerical and smoke tests.

## Packaging

Project metadata and launch entry points are declared in `pyproject.toml`. The installed commands are `dekonwolucje` and `dekonwolucje-gui`. PyTorch is an optional dependency because CPU-only algorithms do not require it.

## PDF documentation

`docs/Deconvolution_GUI_and_Methods_EN.pdf` is the consolidated English description of the GUI and numerical methods. Its LaTeX source is retained beside the PDF.
