#!/usr/bin/env python3
"""Automatic parameter selection for Richardson-Lucy through the public API."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from deconv.api import (
        AutoTuneOptions,
        auto_deconvolve,
        disturb_image,
        generate_motion_psf,
        generate_test_image,
        save_grayscale,
    )
except ModuleNotFoundError:  # Allow direct execution from a source checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deconv.api import (
        AutoTuneOptions,
        auto_deconvolve,
        disturb_image,
        generate_motion_psf,
        generate_test_image,
        save_grayscale,
    )


def main(output_dir: Path | str = "auto_richardson_lucy_output") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reference = generate_test_image(width=192, height=128)
    psf = generate_motion_psf(size=21, angle_deg=32.0)
    disturbed = disturb_image(reference, psf, noise_sigma=0.01, seed=7)

    tuning = auto_deconvolve(
        disturbed,
        psf,
        algorithm="Richardson-Lucy",
        reference=reference,
        params={
            "iterations": 20,
            "epsilon": 1e-8,
            "non_negative": True,
            "begin_with_wiener": False,
        },
        auto_options=AutoTuneOptions(
            strategy="quadratic",
            use_torch_equivalent=True,
            tune_boolean=False,
            tune_categorical=False,
        ),
        progress_callback=print,
    )
    if tuning.deconvolution_result is None:
        raise RuntimeError("Auto did not return the final reconstruction.")
    restored = tuning.deconvolution_result.image

    save_grayscale(reference, output_path / "reference.png")
    save_grayscale(psf.kernel / np.max(psf.kernel), output_path / "motion_psf.png")
    save_grayscale(disturbed, output_path / "disturbed.png")
    save_grayscale(restored, output_path / "restored_auto_rl.png")

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = (
        (reference.data, "Reference"),
        (psf.kernel / psf.kernel.max(), "Oblique motion PSF"),
        (disturbed.data, "Disturbed image"),
        (restored.data, "Auto-tuned Richardson-Lucy"),
    )
    for axis, (array, title) in zip(axes, panels):
        axis.imshow(array, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("Accepted parameters:")
    for name, value in sorted(tuning.best_params.items()):
        print(f"  {name}: {value}")
    print(tuning.status)
    print(f"Saved example outputs in: {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="auto_richardson_lucy_output",
        help="Directory in which PNG outputs are written.",
    )
    arguments = parser.parse_args()
    main(arguments.output_dir)
