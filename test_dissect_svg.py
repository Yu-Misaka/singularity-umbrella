import json
import tempfile
import unittest
from pathlib import Path

from dissect_svg import dissect_svg_to_directory
from preprocess import load_curve_fit


class DissectSvgTests(unittest.TestCase):
    """Regression tests for splitting and filtering multi-subpath SVG input."""

    def test_dissect_svg_exports_filtered_parts_and_curve_fits(self) -> None:
        svg_text = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <path d="M 10,50 C 20,20 40,20 50,50 M 60,60 L 90,90 M 5,5 L 6,6" fill="none" stroke="black"/>
</svg>
"""
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)
            svg_path = tempdir_path / "multi.svg"
            svg_path.write_text(svg_text, encoding="utf-8")

            manifest = dissect_svg_to_directory(
                svg_path,
                output_dir=tempdir_path / "out",
                num_samples=64,
                target_extent=5.0,
                min_svg_length=8.0,
                min_normalized_extent=0.03,
                max_svg_length=100.0,
            )

            self.assertEqual(manifest["source_subpath_count"], 3)
            self.assertEqual(manifest["accepted_part_count"], 2)
            self.assertEqual(manifest["rejected_part_count"], 1)
            self.assertTrue(Path(manifest["manifest_path"]).exists())
            self.assertTrue(Path(manifest["overview_path"]).exists())

            manifest_data = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest_data["parts"]), 2)
            rejected_reasons = {item["reason"] for item in manifest_data["rejected"]}
            self.assertEqual(rejected_reasons, {"too_short"})

            first_fit = load_curve_fit(manifest_data["parts"][0]["curve_fit_path"])
            self.assertEqual(first_fit.num_samples, 64)
            self.assertEqual(first_fit.target_extent, 5.0)


if __name__ == "__main__":
    unittest.main()
