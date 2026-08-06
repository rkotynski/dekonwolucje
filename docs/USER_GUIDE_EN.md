# User guide

## 1. Starting the application

Run `dekonwolucje`, `python -m deconv`, or `python run_deconvolution_gui.py`. The application stores its default settings under the user configuration directory (`~/.config/dekonwolucje` on Linux when `XDG_CONFIG_HOME` is not set). A different JSON profile can be selected from the **Settings** menu; the last selected profile is remembered.

Use **Language → English/Polish** to switch the complete GUI language. The change is immediate and does not alter images, parameters or algorithm profiles.

## 2. Tab 1 — Image and PSF

The first tab loads or generates the source image and PSF. Its previews and histograms show the current calculation data, not an obsolete source snapshot.

Buttons are ordered as follows:

1. **Load disturbed image**
2. **Load reference image** (optional)
3. **Load PSF**
4. **Generate test image**
5. **Generate selected PSF**
6. **Generate degraded input**
7. **Clear images**

For experimental data, **Load disturbed image** creates only the reconstruction input. It deliberately does not create or duplicate a reference image. An independent ground-truth image can be added separately with **Load reference image**; it is used only for PSNR/SSIM and reference-based Auto criteria, never as the reconstruction input. When no reference is loaded, the reference preview remains hidden and reference-based metrics are disabled.

**Clear images** removes loaded/generated images, PSFs, reconstruction histories and results, but preserves the current calculation, GUI and algorithm settings.

If image and PSF arrays have different dimensions, the smaller array is centered on a common canvas by zero padding; pixels are not resampled or cropped.

## 3. Tab 2 — Thresholds and calculation PSF

Tab 2 defines the data actually passed to every algorithm.

- The image floor zeros low-valued input pixels and rescales the surviving range.
- The PSF floor is relative to the PSF peak.
- The red rectangle selects a possibly rectangular PSF window.
- Drag the rectangle to move it, drag an edge or corner to resize it, or use the mouse wheel to scale it around its center.
- **Apply thresholds / PSF selection now** commits the pending controls.
- **Reset thresholds / PSF selection** restores the loaded or generated source arrays.

After Apply, values outside the accepted PSF rectangle are zero in the full-array preview. The selected compact kernel is cropped, projected to nonnegative values and normalized so that its sum equals one. Histograms change only after Apply.

**Optimize PSF floor + Wiener K** jointly proposes a PSF floor and Wiener regularization. With a reference image it minimizes reconstruction MSE. Without a reference, the admissible floor is constrained by robust PSF-background statistics and GCV selects K for each fixed PSF candidate.

## 4. Tab 3 — Algorithm

Select an algorithm and its parameters. Optional processing stages are explicit. Auto tuning freezes the activation state of optional stages at the beginning of a run: parameters belonging to a disabled Wiener initialization, denoiser, TV step or Rosen relaxation are not modified.

The **PyTorch batch** option selects the batched implementation when available. Torch computations use `float32` by default; CUDA is used only when requested and available.

Blind algorithms take the width and height of the estimated PSF directly from the rectangle in Tab 2.

### Block Kaczmarz in practice

The **Block Kaczmarz** method is an experimental ART-style deconvolution. It divides the measured-image plane into square blocks, computes residuals only in selected blocks, combines them with optional overlap and smooth weighting, and back-projects the combined residual through the adjoint PSF operator. It does not explicitly construct the convolution matrix and it is not an exact block projection.

Recommended starting point:

- keep **Full sweep**, **Overlapping blocks**, **Shifted grid**, **Smooth block window**, and **Stabilized sweep** enabled;
- start with block size 32, relaxation 0.15, damping 0.5, and maximum update fraction 0.25;
- reduce relaxation or damping when consecutive frames alternate between overly dark and overly bright results;
- increase block size when seams or local inconsistencies dominate; decrease it when more local correction is desired;
- compare the selected best iteration rather than assuming the last iteration is optimal.

The block count is used only when **Full sweep** is disabled. Randomized order can reduce systematic block-order bias, whereas shifted grids reduce fixed vertical and horizontal boundaries. Optional TV and denoising are applied after each outer Kaczmarz update.

## 5. Tab 4 — Test and result history

Run deconvolution and browse stored iterations. Criteria for all frames are postprocessed in batches; Torch/CUDA is used where available. After a run, the best frame is selected automatically and **Auto levels** is applied.

Black and white sliders cover the full normalized display range and maintain a nonzero gap. They change visualization only, not stored numerical results. **Select best iteration** repeats the automatic frame selection without rerunning the algorithm.

## 6. Wiener implementation

Every Wiener stage uses explicit FFT/IFFT operations. The returned image is the real part of the inverse FFT. The obsolete absolute-value IFFT output and stored degradation-PSF snapshot are not available.

## 7. Auto cancellation

**Cancel Auto** first requests cooperative cancellation. If the current numerical iteration does not return within five seconds, the isolated Auto process is terminated while the GUI process remains alive.

## 8. Using the algorithms without the GUI

The public Qt-independent API is documented in `docs/API_EN.md`. The complete example `examples/wiener_motion_blur.py` generates the standard test image, an oblique motion PSF, a disturbed input, and a Wiener reconstruction.

## AI-assisted development disclosure

Parts of the software and documentation were prepared with assistance from tools based on large language models (LLMs). Their suggestions were incorporated as part of the development process; numerical methods, implementation details and results should nevertheless be independently verified for the intended scientific application.


## Suggested screenshots

Place the following files in `docs/screenshots/`; the LaTeX PDF source inserts them automatically and otherwise displays placeholders.

1. `01-gui-overview.png` — **Main application workflow.** The main window with representative image/PSF data, all four tabs, and optionally the Language menu open.
2. `02-psf-preparation.png` — **Preparation of the calculation PSF.** Tab 2 in full-array view with both histograms and the red rectangular support.
3. `03-kaczmarz-settings.png` — **Block Kaczmarz controls.** Tab 3 showing geometry, sweep/order, stabilization, damping, and update-limit parameters.
4. `04-result-history.png` — **Iteration assessment.** Tab 4 after a multi-iteration run, with criteria, best iteration, and display levels visible.

Use PNG where possible, crop away desktop elements, avoid confidential file paths, and capture the same representative dataset in all figures. A width of roughly 1600–2200 pixels is sufficient for the PDF.
