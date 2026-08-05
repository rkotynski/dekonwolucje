# Python API

The numerical algorithms can be used without PyQt5 and without creating the GUI. The public interface is available from `deconv.api` and is also re-exported from the top-level `deconv` package.

## Installation

From a source checkout:

```bash
python -m pip install -e .
```

Optional Torch/CUDA algorithms require an appropriate PyTorch installation:

```bash
python -m pip install -e ".[gpu]"
```

The public API itself does not import Qt. CPU algorithms therefore work in scripts, notebooks, services, and headless batch jobs.

## Main data objects

- `GrayImage`: a two-dimensional grayscale image. Numerical values are available in `image.data`.
- `PSF`: a non-negative, unit-sum point-spread function. The calculation kernel is available in `psf.kernel`.
- `DeconvolutionResult`: reconstruction result containing `image`, `history`, `metrics`, and `info`.

For a result returned by an iterative method:

```python
restored_array = result.image.data
iteration_arrays = [frame.data for frame in result.history]
```

Blind methods additionally store the final estimated PSF in:

```python
estimated_psf = result.image.metadata.get("estimated_psf")
```

## Public functions

### Data conversion and generation

```python
as_gray_image(array, name="image", normalize=False)
as_psf(array, name="psf")
generate_test_image(width=256, height=None, padding=0)
generate_motion_psf(size=21, angle_deg=35.0)
```

`as_gray_image(..., normalize=False)` preserves input intensities and requires values in `[0, 1]`. Set `normalize=True` to apply min-max normalization. `as_psf()` clips negative values and normalizes the kernel to unit sum.

`generate_test_image()` uses the same synthetic-image generator as the GUI. `generate_motion_psf()` creates a horizontal or oblique motion kernel.

### Forward disturbance model

```python
disturbed = disturb_image(
    image,
    psf,
    noise_sigma=0.01,
    noise_type="Gaussian",
    seed=7,
)
```

The function uses the same zero-boundary linear `same` convolution as the GUI. Supported disturbance names are the same as in the application. A seed makes synthetic noise reproducible.

### Algorithm discovery

```python
names = available_algorithms()
params = default_parameters("Richardson-Lucy")
```

Algorithm names are canonical English registry identifiers and do not depend on the GUI language.

### General execution

```python
result = run_deconvolution(
    disturbed,
    psf,
    algorithm="Richardson-Lucy",
    iterations=40,
    epsilon=1e-8,
    non_negative=True,
)
```

Alternatively, pass a parameter mapping:

```python
result = run_deconvolution(
    disturbed,
    psf,
    algorithm="Block Kaczmarz",
    params={
        "iterations": 20,
        "kaczmarz_block_size": 32,
        "kaczmarz_relaxation": 0.15,
    },
)
```

Keyword parameters override values supplied through `params`, and both override the algorithm defaults.

### Wiener convenience wrapper

```python
result = wiener_filter(
    disturbed,
    psf,
    K=2e-3,
    non_negative=True,
)
```

This calls the same explicit FFT/IFFT Wiener implementation as the GUI. The returned image is the real part of the inverse FFT.

### Saving results

```python
save_grayscale(result.image, "restored.png")
```

The helper writes an 8-bit grayscale image. For quantitative work, preserve the floating-point array separately, for example with NumPy or a MAT file.

## Complete example

The repository contains:

```text
examples/wiener_motion_blur.py
```

Run it from the repository root after installation:

```bash
python examples/wiener_motion_blur.py --output-dir wiener_motion_output
```

It generates the GUI test image, an oblique 31-pixel motion PSF, a reproducibly disturbed image, and a Wiener reconstruction. The output directory contains:

- `reference.png`
- `motion_psf.png`
- `disturbed.png`
- `restored_wiener.png`
- `comparison.png`

## NumPy-array example

```python
import numpy as np
from deconv.api import run_deconvolution

measured = np.load("measured.npy")       # 2D values in [0, 1]
psf = np.load("psf.npy")                 # non-negative 2D kernel

result = run_deconvolution(
    measured,
    psf,
    algorithm="Wiener",
    K=1e-3,
)
restored = result.image.data
```

## Boundary models and PSF preparation

The API passes the same calculation-safe PSF to the algorithms:

1. the input PSF is projected to non-negative values;
2. it is normalized to unit sum;
3. if it is larger than the image, it is centrally fitted to the image grid.

The closed-form Wiener algorithm uses circular FFT convolution. Most iterative methods use zero-boundary linear convolution and its exact adjoint. Consequently, border behavior may differ even for the same image and PSF.

## Progress and cooperative stopping

Iterative algorithms accept the internal-compatible keyword parameters:

```python
_iteration_callback=current_iteration_callback
_stop_event=threading_or_multiprocessing_event
```

The callback receives `(current, total)`. When the event is set, cooperative algorithms stop after the current iteration. These names begin with an underscore because they are advanced execution hooks and may evolve more readily than the core API.

## API stability

The stable entry points are the functions exported by `deconv.api` and the public fields of `GrayImage`, `PSF`, and `DeconvolutionResult`. GUI classes and objects in `legacy_runtime.py` are not part of the public numerical API.
