import json
import tempfile
import unittest
from pathlib import Path

from dissect_svg import dissect_svg_to_directory
from export_system import export_system_from_curve_fit, export_systems_from_manifest
from preprocess_svg import fit_curve_from_svg


class ExportSystemTests(unittest.TestCase):
    """Regression tests for serializing reconstructible system definitions."""

    def test_export_system_from_curve_fit_writes_geometry_and_dynamics(self) -> None:
        svg_text = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <path d="M 10,70 C 25,25 45,25 60,70 C 45,62 25,62 10,70" fill="none" stroke="black"/>
</svg>
"""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            svg_path = root / "curve.svg"
            fit_path = root / "curve_fit.json"
            svg_path.write_text(svg_text, encoding="utf-8")
            fit_curve_from_svg(svg_path, num_samples=64, target_extent=5.0).save(fit_path)

            output = export_system_from_curve_fit(
                fit_path,
                output_path=root / "system.json",
                model_kwargs={"total_time": 10.0, "transient_time": 2.0, "dt": 0.01},
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"]["curve_fit_path"], str(fit_path))
            self.assertEqual(payload["source"]["num_samples"], 64)
            self.assertIn("embedding", payload)
            self.assertIn("dynamics", payload)
            self.assertGreater(len(payload["embedding"]["centerline"]), 0)
            self.assertIn("track_fraction", payload["embedding"])

    def test_export_systems_from_manifest_writes_index(self) -> None:
        svg_text = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <path d="M 10,70 C 25,25 45,25 60,70 M 62,60 C 75,35 88,35 92,58" fill="none" stroke="black"/>
</svg>
"""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            svg_path = root / "multi.svg"
            svg_path.write_text(svg_text, encoding="utf-8")
            manifest = dissect_svg_to_directory(
                svg_path,
                output_dir=root / "dissected",
                num_samples=48,
                target_extent=5.0,
                min_svg_length=8.0,
                min_normalized_extent=0.02,
                max_svg_length=120.0,
            )

            result = export_systems_from_manifest(
                manifest["manifest_path"],
                output_dir=root / "systems",
                limit_parts=1,
                model_kwargs={"total_time": 10.0, "transient_time": 2.0, "dt": 0.01},
            )

            index_path = Path(result["index_path"])
            self.assertTrue(index_path.exists())
            self.assertEqual(result["exported_part_count"], 1)
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["parts"]), 1)
            self.assertTrue(Path(payload["parts"][0]["system_path"]).exists())


if __name__ == "__main__":
    unittest.main()
