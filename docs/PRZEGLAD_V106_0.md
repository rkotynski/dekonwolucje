# Dekonwolucje 0.106.0 - Auto in the public API

- Public functions `auto_tune_parameters()` and `auto_deconvolve()`.
- Typed `AutoTuneOptions` and `AutoTuningResult`.
- Quadratic-coordinate and bounded full-batched search.
- PSNR scoring with an independent reference, Wiener GCV without a reference, and the GUI no-reference criterion for other methods.
- Frozen disabled optional stages.
- Optional Torch-batched proxy tuning followed by validation on the requested algorithm.
- Progress callback and cooperative stop event.
- Example: `examples/auto_richardson_lucy_motion.py`.
