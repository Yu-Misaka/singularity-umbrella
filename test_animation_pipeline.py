import json
import py_compile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dissect_svg import dissect_svg_to_directory
from export_blender_paths import export_blender_paths
from schedule_parts import build_schedule


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return float(np.linalg.norm(point - polyline[0]))
    best = float("inf")
    for start, end in zip(polyline[:-1], polyline[1:], strict=False):
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            candidate = float(np.linalg.norm(point - start))
        else:
            weight = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
            projection = start + weight * segment
            candidate = float(np.linalg.norm(point - projection))
        best = min(best, candidate)
    return best


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
                frames_per_unit=12.0,
            )

            paths_payload = json.loads(paths_path.read_text(encoding="utf-8"))
            schedule_payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            self.assertGreater(paths_payload["part_count"], 0)
            self.assertEqual(len(paths_payload["parts"]), len(schedule_payload["parts"]))
            self.assertIn("world", paths_payload)
            first_part = paths_payload["parts"][0]
            self.assertGreater(len(first_part["orbit_points_world"]), 10)
            self.assertGreater(len(first_part["guided_points_world"]), 10)
            self.assertGreater(first_part["orbit_arc_length_total_world"], 0.0)
            self.assertEqual(len(first_part["orbit_points_world"]), len(first_part["orbit_arc_length_samples_world"]))
            self.assertLess(first_part["guided_segment"]["start_arc_length_world"], first_part["guided_segment"]["end_arc_length_world"])
            orbit = np.asarray(first_part["orbit_points_world"], dtype=float)
            entry = np.asarray(first_part["guided_entry_point_world"], dtype=float)
            exit = np.asarray(first_part["guided_exit_point_world"], dtype=float)
            self.assertLess(_distance_to_polyline(entry, orbit), 1e-6)
            self.assertLess(_distance_to_polyline(exit, orbit), 1e-6)

            enter_frames = {part["enter_target_frame"] for part in schedule_payload["parts"]}
            self.assertEqual(len(enter_frames), 1)
            self.assertEqual(next(iter(enter_frames)), schedule_payload["align_frame"])
            self.assertEqual(schedule_payload["scene_frame_start"], 0)
            travel_frames = {part["travel_frames"] for part in schedule_payload["parts"]}
            self.assertGreater(len(travel_frames), 1)
            self.assertEqual(min(part["start_frame"] for part in schedule_payload["parts"]), 0)
            for part in schedule_payload["parts"]:
                self.assertGreaterEqual(part["entry_progress"], 0.0)
                self.assertLessEqual(part["exit_progress"], 1.0)
                self.assertLess(part["entry_progress"], part["exit_progress"])

    def test_blender_import_script_compiles(self) -> None:
        py_compile.compile("blender_import.py", doraise=True)
        source = Path("blender_import.py").read_text(encoding="utf-8")
        self.assertIn("enter_target_event", source)
        self.assertIn("inside_target_window", source)
        self.assertIn("exit_target_event", source)
        self.assertIn("GeometryNodeSampleCurve", source)
        self.assertIn("GeometryNodeInstanceOnPoints", source)
        self.assertNotIn("primitive_uv_sphere_add", source)


if __name__ == "__main__":
    unittest.main()
