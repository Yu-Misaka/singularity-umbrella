import json
import tempfile
import unittest
from pathlib import Path

from batch_compose import compose_manifest
from dissect_svg import dissect_svg_to_directory


class BatchComposeTests(unittest.TestCase):
    """Regression tests for batch composition over dissected SVG parts."""

    def test_compose_manifest_builds_composite_and_report(self) -> None:
        svg_text = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <path d="M 10,70 C 25,25 45,25 60,70 M 65,25 L 90,60" fill="none" stroke="black"/>
</svg>
"""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            svg_path = root / "multi.svg"
            svg_path.write_text(svg_text, encoding="utf-8")
            manifest = dissect_svg_to_directory(
                svg_path,
                output_dir=root / "dissected",
                num_samples=64,
                target_extent=5.0,
                min_svg_length=8.0,
                min_normalized_extent=0.02,
                max_svg_length=120.0,
            )

            result = compose_manifest(
                manifest["manifest_path"],
                output_path=root / "composite.png",
                report_path=root / "report.json",
                image_size=(900, 900),
                model_kwargs={
                    "total_time": 10.0,
                    "transient_time": 2.0,
                    "dt": 0.01,
                },
            )

            self.assertTrue(result.output_path.exists())
            self.assertTrue(result.report_path.exists())
            payload = json.loads(result.report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["accepted_part_count"], result.accepted_part_count)
            self.assertGreater(result.accepted_part_count, 0)
            self.assertEqual(len(payload["parts"]), result.accepted_part_count)
            self.assertLess(result.mean_projection_error, 0.8)
            self.assertLess(result.max_projection_error, 1.1)


if __name__ == "__main__":
    unittest.main()
