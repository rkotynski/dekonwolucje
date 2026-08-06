## 0.108.0

- Added a bilingual **Clear images** action in Tab 1.
- The action removes loaded/generated images, PSFs, reconstruction histories and results while preserving GUI and algorithm settings.
- Clearing is postponed safely if an Auto or reconstruction worker has not yet stopped.

## 0.107.0

- Separated experimental disturbed-image loading from optional reference-image loading in Tab 1.
- Loading a disturbed image no longer creates or duplicates a reference image.
- Added a dedicated **Load reference image** action for optional ground truth used only by metrics and reference-based Auto.
- Preserved separately loaded disturbed and reference images when PSF or zero-padding settings are changed.
- Added regression tests and updated bilingual user/PDF documentation.

## 0.106.0

- Added GUI-independent Auto parameter selection through `auto_tune_parameters()` and `auto_deconvolve()`.
- Added `AutoTuneOptions`, `AutoTuningResult`, quadratic-coordinate and full-batched search strategies.
- Preserved the GUI rule that disabled Wiener, denoiser and optional TV stages cannot be changed by Auto.
- Added reference-based PSNR scoring, reference-free Wiener GCV and the GUI no-reference criterion.
- Added a standalone Auto-tuned Richardson-Lucy example and bilingual API/PDF documentation.

## 0.105.0

- Expanded the Qt-independent API to expose all 15 registered algorithms through the generic runner and dedicated convenience wrappers.
- Added Gaussian, high-frequency, and incoherent-lens PSF generators plus a generic PSF dispatcher.
- Added and corrected standalone examples for Richardson-Lucy, Richardson-Lucy-Wiener, Richardson-Lucy-Rosen, Landweber, and Block Kaczmarz.
- Updated project authors to Amine Güneş and Rafał Kotyński, University of Warsaw, Faculty of Physics.

## 0.104.4

- Embedded the four final GUI screenshots directly in the main bilingual README.
- Added bilingual captions describing the main workflow, PSF preparation, Block Kaczmarz controls and iteration assessment.
- Removed README wording about screenshot placeholders.

## 0.104.3

- Expanded the Richardson-Lucy-Rosen documentation to show how the nonlinear spectral correlation replaces the classical Richardson-Lucy adjoint back-projection.
- Defined the distinction between elementwise spectral conjugation `H*` and the adjoint operator `\mathcal H*`.
- Clarified that `L=M=1` gives circular correlation before normalization, but is not exactly the zero-boundary linear-same Richardson-Lucy update used elsewhere in the program.

## 0.104.2

- Expanded the PDF definition of the zero-boundary linear convolution and related the operator notation explicitly to the calculation PSF.
- Defined the two-dimensional DFT pair exactly as used by SciPy and PyTorch (`norm="backward"`) and documented the circular-convolution theorem and wrap-around boundary condition.
- Moved the AI-assisted development disclosure to the end of the PDF.

## 0.104.1

- Added the four supplied GUI screenshots to the repository and English PDF.
- Replaced screenshot placeholders with final captions tied to the actual views.
- Added clickable original/classical references for the documented numerical methods.
- Clarified which hybrid algorithms and stabilization choices are implementation-specific.

## 0.103.3

## 0.104.0

- Added a Qt-independent public Python API for all registered algorithms.
- Added conversion, synthetic-image, motion-PSF, disturbance, saving, and Wiener convenience functions.
- Added a complete standalone Wiener example with an oblique motion PSF.
- Added bilingual API documentation and an API section to the PDF.

- Expanded the mathematical and implementation documentation of the Block Kaczmarz method.
- Added drop-in screenshot placeholders and a proposed figure set with captions.
- Added bilingual practical Kaczmarz guidance.
- Corrected GitHub Actions compatibility for Python 3.10 and optional PyTorch tests.

# Changelog / Historia zmian

- [English changelog](docs/CHANGELOG_EN.md)
- [Historia zmian po polsku](docs/CHANGELOG_PL.md)

Detailed notes for earlier development versions are retained in [`docs/history/`](docs/history/).

Szczegółowe notatki dotyczące wcześniejszych wersji rozwojowych zachowano w katalogu [`docs/history/`](docs/history/).
