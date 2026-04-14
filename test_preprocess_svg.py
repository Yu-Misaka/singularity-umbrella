import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import build_guided_chaotic_field
from preprocess import load_curve_fit
from preprocess_svg import fit_curve_from_svg
from visual import render_comparison_image


WORKDIR = Path("/Users/suzuka/Documents/Mason/singularity-umbrella")
TEST_SVG = WORKDIR / "experiment_outputs" / "test.svg"


class PreprocessSvgTests(unittest.TestCase):
    def test_fit_curve_from_svg_returns_scaled_samples(self) -> None:
        fit = fit_curve_from_svg(TEST_SVG, num_samples=140)
        samples = fit.as_array()
        self.assertEqual(samples.shape, (140, 2))
        self.assertTrue(np.isfinite(samples).all())
        self.assertEqual(fit.threshold, -1)
        self.assertAlmostEqual(float(max(np.ptp(samples[:, 0]), np.ptp(samples[:, 1]))), 6.0, places=6)

    def test_save_and_load_roundtrip(self) -> None:
        fit = fit_curve_from_svg(TEST_SVG, num_samples=96, target_extent=5.0)
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "curve_fit_svg.json"
            fit.save(output)
            loaded = load_curve_fit(output)
            self.assertEqual(loaded.num_samples, 96)
            self.assertEqual(loaded.target_extent, 5.0)
            np.testing.assert_allclose(loaded.as_array(), fit.as_array())

    def test_end_to_end_pipeline_from_svg_to_render(self) -> None:
        fit = fit_curve_from_svg(TEST_SVG, num_samples=140)
        model = build_guided_chaotic_field(fit.as_array(), num_curve_samples=fit.num_samples)
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "comparison_svg.png"
            comparison = render_comparison_image(model, output_path=output)
            self.assertTrue(output.exists())
            self.assertLess(comparison.max_projection_error, 0.05)
            self.assertLess(comparison.mean_projection_error, 0.03)


if __name__ == "__main__":
    unittest.main()
