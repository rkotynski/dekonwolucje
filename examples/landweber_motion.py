#!/usr/bin/env python3
"""Standalone Landweber deconvolution example using the public deconv API.

The script generates the same synthetic test image as the GUI, creates an
oblique motion PSF, forms a disturbed image with reproducible Gaussian noise,
applies the iterative Landweber algorithm, and saves both individual images
and a comparison figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from deconv.api import (
        disturb_image,
        generate_motion_psf,
        generate_test_image,
        landweber_filter,
        save_grayscale,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deconv.api import (
        disturb_image,
        generate_motion_psf,
        generate_test_image,
        landweber_filter,
        save_grayscale,

    )


def main(output_dir: Path | str = "landweber_motion_output") -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reference = generate_test_image(width=384, height=256)

    psf = generate_motion_psf(
        size=31,
        angle_deg=30.0,
    )

    disturbed = disturb_image(
        reference,
        psf,
        noise_sigma=0.01,
        noise_type="Gaussian",
        seed=7,
    )

    result = landweber_filter(
        disturbed,
        psf,
        iterations=50,
        step=0.8,
        non_negative=True,
        begin_with_wiener=True,
    )

    restored = result.image

    save_grayscale(reference, output_path / "reference.png")
    save_grayscale(psf.kernel / np.max(psf.kernel), output_path / "motion_psf.png")
    save_grayscale(disturbed, output_path / "disturbed.png")
    save_grayscale(restored, output_path / "restored_landweber.png")

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))

    panels = (
        (reference.data, "Reference"),
        (psf.kernel / psf.kernel.max(), "Motion PSF"),
        (disturbed.data, "Disturbed image"),
        (restored.data, "Landweber restoration"),
    )

    for axis, (array, title) in zip(axes, panels):
        axis.imshow(array, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(
        output_path / "comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(result.info)
    print(f"Saved example outputs in: {output_path.resolve()}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="landweber_motion_output",
        help="Directory in which PNG outputs are written.",
    )

    arguments = parser.parse_args()
    main(arguments.output_dir)