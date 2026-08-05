from __future__ import annotations

import unittest
import numpy as np

from deconv.algorithms.registry import AlgorithmRegistry
from deconv.core.runtime import TORCH_AVAILABLE, GrayImage, PSF, degrade_image, degradation_kernel_for_image


class AlgorithmSmokeTests(unittest.TestCase):
    def test_registered_algorithms_return_finite_images(self) -> None:
        reference = GrayImage.synthetic(48, 48, padding=6)
        psf = degradation_kernel_for_image(PSF.gaussian(9, 1.7), reference.data.shape, max_width=9)
        measured = degrade_image(reference, psf, noise_sigma=0.0)
        common = dict(
            iterations=2,
            prefer_cuda=False,
            torch_float64=False,
            non_negative=True,
            K=0.01,
            epsilon=1e-7,
            blind_psf_height=5,
            blind_psf_width=7,
            torch_record_every=1,
            kaczmarz_block_size=12,
            kaczmarz_blocks_per_iteration=4,
        )
        registry = AlgorithmRegistry()
        for name in registry.names():
            with self.subTest(algorithm=name):
                if name.startswith(("Torch ", "PyTorch ")) and not TORCH_AVAILABLE:
                    continue
                result = registry.get(name).run(measured, psf, **common)
                self.assertEqual(result.image.data.shape, measured.data.shape)
                self.assertTrue(np.isfinite(result.image.data).all())
                if name in {"Blind Richardson-Lucy", "PyTorch Blind Adam TV-MAP"}:
                    estimated = np.asarray(result.image.metadata.get("estimated_psf"))
                    self.assertEqual(estimated.shape, (5, 7))
                    self.assertAlmostEqual(float(estimated.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
