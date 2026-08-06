#!/usr/bin/env python3
"""Block Kaczmarz example with a Gaussian PSF using the public API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from deconv.api import (
        disturb_image,
        generate_gaussian_psf,
        generate_test_image,
        block_kaczmarz_filter,
        save_grayscale,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deconv.api import (
        disturb_image,
        generate_gaussian_psf,
        generate_test_image,
        block_kaczmarz_filter,
        save_grayscale,
    )


def main(output_dir: Path | str = "block_kaczmarz_output") -> Path:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reference = generate_test_image(width=384, height=256)

    psf = generate_gaussian_psf(
        size=21,
        sigma=3,
    )

    disturbed = disturb_image(
        reference,
        psf,
        noise_sigma=0.01,
        noise_type="Gaussian",
        seed=7,
    )

    result = block_kaczmarz_filter(
        disturbed,
        psf,
        iterations=20,
        relaxation=0.15,
        block_size=32,
        full_sweep=True,
        overlap=True,
        randomized=False,
        begin_with_wiener=False,
    )

    restored = result.image

    save_grayscale(reference, output_path / "reference.png")
    save_grayscale(psf.kernel / np.max(psf.kernel), output_path / "gaussian_psf.png")
    save_grayscale(disturbed, output_path / "disturbed.png")
    save_grayscale(restored, output_path / "restored_block_kaczmarz.png")

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))

    panels = (
        (reference.data, "Reference"),
        (psf.kernel / psf.kernel.max(), "Gaussian PSF"),
        (disturbed.data, "Disturbed image"),
        (restored.data, "Block Kaczmarz"),
    )

    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

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
        default="block_kaczmarz_output",
        help="Directory in which PNG outputs are written.",
    )

    args = parser.parse_args()
    main(args.output_dir)