import json
import py_compile
import tempfile
import unittest
from pathlib import Path

from dissect_svg import dissect_svg_to_directory
from export_blender_paths import export_blender_paths
from schedule_parts import build_schedule


class AnimationPipelineTests(unittest.TestCase):
    """Regression tests for Blender-oriented path export and scheduling."""

    def test_export_blender_paths_and_schedule(self) -> None:
        svg_text = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <path d="M 10,70 C 25,25 45,25 60,70 C 45,62 25,62 10,70 M 68,60 C 76,35 86,35 92,58" fill="none" stroke="black"/>
</svg>
"""
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            svg_path = root / "portrait.svg"
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

            paths_path = export_blender_paths(
                manifest["manifest_path"],
                output_path=root / "blender_paths.json",
                orbit_sample_count=160,
                guided_sample_count=60,
                target_sample_count=60,
                model_kwargs={"total_time": 10.0, "transient_time": 2.0, "dt": 0.01},
            )
            schedule_path = build_schedule(
                paths_path,
                output_path=root / "schedule.json",
                fps=24,
                align_frame=120,
                travel_frames=180,
            )

            paths_payload = json.loads(paths_path.read_text(encoding="utf-8"))
            schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            self.assertGreater(paths_payload["part_count"], 0)
            self.assertEqual(len(paths_payload["parts"]), len(schedule_payload["parts"]))
            self.assertIn("world", paths_payload)
            first_part = paths_payload["parts"][0]
            self.assertGreater(len(first_part["orbit_points_world"]), 10)
            self.assertGreater(len(first_part["guided_points_world"]), 10)
            self.assertAlmostEqual(first_part["guided_segment"]["mid_factor"], schedule_payload["parts"][0]["guided_mid_factor"], places=6)

    def test_blender_import_script_compiles(self) -> None:
        py_compile.compile("blender_import.py", doraise=True)


if __name__ == "__main__":
    unittest.main()
