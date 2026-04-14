import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import build_guided_chaotic_field
from preprocess import fit_curve_from_image, load_curve_fit
from visual import render_comparison_image


WORKDIR = Path("/Users/suzuka/Documents/Mason/singularity-umbrella")
TEST_IMAGE = WORKDIR / "test.jpg"


class PreprocessTests(unittest.TestCase):
    def test_fit_curve_from_image_returns_normalized_samples(self) -> None:
        fit = fit_curve_from_image(TEST_IMAGE, num_samples=140)
        samples = fit.as_array()
        self.assertEqual(samples.shape, (140, 2))
        self.assertTrue(np.isfinite(samples).all())
        self.assertEqual(fit.image_size, (256, 256))
        self.assertAlmostEqual(float(max(np.ptp(samples[:, 0]), np.ptp(samples[:, 1]))), 6.0, places=6)
        self.assertLess(abs(float(np.mean(samples[:, 0]))), 1e-6)
        self.assertLess(abs(float(np.mean(samples[:, 1]))), 1e-6)

    def test_save_and_load_curve_fit_roundtrip(self) -> None:
        fit = fit_curve_from_image(TEST_IMAGE, num_samples=96, target_extent=5.0)
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "curve_fit.json"
            fit.save(output)
            loaded = load_curve_fit(output)
            self.assertEqual(loaded.num_samples, 96)
            self.assertEqual(loaded.target_extent, 5.0)
            np.testing.assert_allclose(loaded.as_array(), fit.as_array())

    def test_end_to_end_pipeline_from_image_to_render(self) -> None:
        fit = fit_curve_from_image(TEST_IMAGE, num_samples=140)
        model = build_guided_chaotic_field(fit.as_array(), num_curve_samples=fit.num_samples)
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "comparison.png"
            comparison = render_comparison_image(model, output_path=output)
            self.assertTrue(output.exists())
            self.assertLess(comparison.max_projection_error, 0.05)
            self.assertLess(comparison.mean_projection_error, 0.03)


if __name__ == "__main__":
    unittest.main()
