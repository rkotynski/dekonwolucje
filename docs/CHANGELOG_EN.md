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

# v103.2 - translation audit and documentation disclosure

- Audited static and dynamic GUI text and completed missing Polish translations.
- Replaced phrase-fragment substitution for dynamic messages with full message-template translation, preventing mixed-language and corrupted words.
- Standardized the Polish GUI term for a degraded input as **obraz zaburzony**.
- Added an LLM-assisted-development disclosure to the project description and documentation.
- Removed author information from the English PDF documentation.

# v103.1 - startup hotfix and PDF documentation

- Added the missing `sys` import required by platform-specific configuration-directory selection.
- Added an English PDF describing the GUI, data flow, convolution models, implemented methods, metrics, Auto tuning, and software architecture.
- Included the PDF in the GitHub source package and installed wheel documentation.

# Changelog

## v103 — bilingual GUI and GitHub-ready project

- Added complete English/Polish GUI switching, persisted in the active JSON settings profile.
- Added translated widget wrappers for form labels, buttons, checkboxes, combo-box displays, tooltips, dialogs, status messages, logs and Matplotlib titles.
- Preserved canonical English algorithm identifiers, configuration keys and source-code comments.
- Added `pyproject.toml`, MIT license, citation metadata, GitHub Actions tests, issue templates and bilingual documentation.
- Moved the default settings file to the user configuration directory and remembered the last selected settings profile.

## v102

- Auto tuning freezes optional stages that were disabled when tuning started.
- Corrected no-reference joint optimization of PSF floor and Wiener K by constraining the floor with robust PSF-background statistics and rejecting collapsed kernels.

## v101

- Added cooperative Auto cancellation with a five-second watchdog and isolated-process termination.

## v100

- Thresholds and the PSF frame become calculation data only after Apply.
- Removed absolute-value inverse-FFT output from Wiener filtering.

## v99 and earlier

Earlier releases introduced one explicit calculation PSF, rectangular PSF selection, batched iteration metrics, fast display levels, consistent FFT/convolution operators and extensive numerical regression tests. Detailed Polish historical notes are retained in `docs/history/`.
