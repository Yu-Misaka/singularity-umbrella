from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import GuidedChaoticField, build_guided_chaotic_field
from preprocess import CurveFit, load_curve_fit


type Array = np.ndarray


@dataclass(slots=True)
class WorldConfig:
    """Global coordinate-system description used by Blender-side importers."""

    canvas_extent: float = 12.0
    canvas_z: float = 0.0
    depth_scale: float = 0.35
    target_extent: float = 6.0
    plane_axes: tuple[str, str] = ("X", "Y")
    projection_axis: str = "-Z"

    def local_to_world_depth_scale(self) -> float:
        return float(self.canvas_extent / max(self.target_extent, 1e-12) * self.depth_scale)


def _viewbox_center(viewbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_x, min_y, width, height = viewbox
    return min_x + 0.5 * width, min_y + 0.5 * height


def _svg_to_world_xy(
    points: Array,
    *,
    viewbox: tuple[float, float, float, float],
    world: WorldConfig,
) -> Array:
    """Map SVG coordinates into a world-space canvas centered at the origin."""

    center_x, center_y = _viewbox_center(viewbox)
    _, _, width, height = viewbox
    uniform_scale = world.canvas_extent / max(width, height, 1e-12)
    result = np.empty((len(points), 2), dtype=float)
    result[:, 0] = (points[:, 0] - center_x) * uniform_scale
    result[:, 1] = (center_y - points[:, 1]) * uniform_scale
    return result


def _local_to_svg_xy(
    points: Array,
    *,
    normalized_center: tuple[float, float],
    fit_scale: float,
    viewbox: tuple[float, float, float, float],
) -> Array:
    """Invert the per-part normalization so local model coordinates return to SVG space."""

    if fit_scale <= 0:
        raise ValueError("fit_scale must be positive")
    min_x, min_y, width, height = viewbox
    center = np.asarray(normalized_center, dtype=float)
    normalized = np.asarray(points, dtype=float) / float(fit_scale) + center
    svg = np.empty_like(normalized)
    svg[:, 0] = min_x + normalized[:, 0] * width
    svg[:, 1] = min_y + (1.0 - normalized[:, 1]) * height
    return svg


def _local_to_world_points(
    local_points: Array,
    *,
    normalized_center: tuple[float, float],
    fit_scale: float,
    viewbox: tuple[float, float, float, float],
    world: WorldConfig,
) -> Array:
    """Convert a local 3D orbit from `core.py` into Blender world coordinates."""

    svg_xy = _local_to_svg_xy(
        local_points[:, :2],
        normalized_center=normalized_center,
        fit_scale=fit_scale,
        viewbox=viewbox,
    )
    world_xy = _svg_to_world_xy(svg_xy, viewbox=viewbox, world=world)
    result = np.empty((len(local_points), 3), dtype=float)
    result[:, :2] = world_xy
    result[:, 2] = world.canvas_z + local_points[:, 2] * world.local_to_world_depth_scale()
    return result


def _downsample_indices(length: int, sample_count: int) -> Array:
    if length < 1:
        raise ValueError("length must be positive")
    if sample_count < 2 or length <= sample_count:
        return np.arange(length, dtype=int)
    return np.unique(np.round(np.linspace(0, length - 1, sample_count)).astype(int))


def _slice_with_indices(points: Array, indices: Array) -> Array:
    return np.asarray(points, dtype=float)[indices]


def _cumulative_arc_lengths(points: Array) -> Array:
    """Return cumulative world-space arc length along a polyline."""

    samples = np.asarray(points, dtype=float)
    if len(samples) == 0:
        raise ValueError("cannot measure arc length of an empty polyline")
    if len(samples) == 1:
        return np.array([0.0], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _sample_polyline_by_arc_length(points: Array, *, sample_count: int) -> tuple[Array, Array]:
    """Resample a polyline uniformly by arc length."""

    samples = np.asarray(points, dtype=float)
    cumulative = _cumulative_arc_lengths(samples)
    total = float(cumulative[-1])
    if sample_count <= 0 or len(samples) <= sample_count or total <= 1e-12:
        return samples.copy(), cumulative

    targets = np.linspace(0.0, total, sample_count, dtype=float)
    result = np.empty((sample_count, samples.shape[1]), dtype=float)
    segment = 0
    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment >= len(samples) - 1:
            result[index] = samples[-1]
            continue
        local_start = cumulative[segment]
        local_length = max(cumulative[segment + 1] - cumulative[segment], 1e-12)
        weight = (target - local_start) / local_length
        result[index] = (1.0 - weight) * samples[segment] + weight * samples[segment + 1]
    return result, targets


def _sample_polyline_at_arc_lengths(points: Array, cumulative: Array, targets: Array) -> Array:
    """Sample a polyline at explicit cumulative arc-length positions."""

    samples = np.asarray(points, dtype=float)
    arc = np.asarray(cumulative, dtype=float)
    query = np.asarray(targets, dtype=float)
    if len(samples) != len(arc):
        raise ValueError("points and cumulative arc lengths must have matching length")
    if len(samples) == 0:
        raise ValueError("cannot sample an empty polyline")
    if len(samples) == 1:
        return np.repeat(samples[:1], len(query), axis=0)

    total = float(arc[-1])
    clamped = np.clip(query, 0.0, total)
    result = np.empty((len(clamped), samples.shape[1]), dtype=float)
    segment = 0
    for index, target in enumerate(clamped):
        while segment + 1 < len(arc) and arc[segment + 1] < target:
            segment += 1
        if segment >= len(samples) - 1:
            result[index] = samples[-1]
            continue
        local_start = arc[segment]
        local_length = max(arc[segment + 1] - arc[segment], 1e-12)
        weight = (target - local_start) / local_length
        result[index] = (1.0 - weight) * samples[segment] + weight * samples[segment + 1]
    return result


def _extract_subpolyline_by_arc_length(
    points: Array,
    cumulative: Array,
    *,
    start_arc: float,
    end_arc: float,
    sample_count: int,
) -> Array:
    """Extract one arc-length window as a resampled sub-polyline."""

    if end_arc < start_arc:
        raise ValueError("end_arc must be greater than or equal to start_arc")
    if sample_count < 2:
        sample_count = 2
    targets = np.linspace(start_arc, end_arc, sample_count, dtype=float)
    return _sample_polyline_at_arc_lengths(points, cumulative, targets)


def _part_report_lookup(batch_report_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if batch_report_path is None:
        return {}
    payload = json.loads(Path(batch_report_path).read_text(encoding="utf-8"))
    return {entry["part_id"]: entry for entry in payload.get("parts", [])}


def _export_part_payload(
    part: dict[str, Any],
    *,
    model: GuidedChaoticField,
    fit: CurveFit,
    viewbox: tuple[float, float, float, float],
    world: WorldConfig,
    orbit_sample_count: int,
    guided_sample_count: int,
    target_sample_count: int,
    batch_report_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    orbit = model.orbit
    orbit_world_full = _local_to_world_points(
        orbit.spatial,
        normalized_center=tuple(part["normalized_center"]),
        fit_scale=float(part["fit_scale"]),
        viewbox=viewbox,
        world=world,
    )
    orbit_cumulative_world = _cumulative_arc_lengths(orbit_world_full)
    orbit_world, orbit_arc_samples_world = _sample_polyline_by_arc_length(
        orbit_world_full,
        sample_count=orbit_sample_count,
    )
    orbit_total_arc_world = float(orbit_cumulative_world[-1])
    orbit_time_targets = orbit_arc_samples_world / max(orbit_total_arc_world, 1e-12) * float(orbit.time[-1])
    target_local = fit.as_array()
    target_indices = _downsample_indices(len(target_local), target_sample_count)

    target_svg = _local_to_svg_xy(
        _slice_with_indices(target_local, target_indices),
        normalized_center=tuple(part["normalized_center"]),
        fit_scale=float(part["fit_scale"]),
        viewbox=viewbox,
    )
    target_world_xy = _svg_to_world_xy(target_svg, viewbox=viewbox, world=world)
    target_world = np.column_stack([target_world_xy, np.full(len(target_world_xy), world.canvas_z, dtype=float)])

    total_frames_factor = max(len(orbit.spatial) - 1, 1)
    guided_mid_index = 0.5 * (model.guided_segment.start_index + model.guided_segment.end_index)
    guided_start_arc_world = float(orbit_cumulative_world[model.guided_segment.start_index])
    guided_end_arc_world = float(orbit_cumulative_world[model.guided_segment.end_index])
    guided_mid_arc_world = 0.5 * (guided_start_arc_world + guided_end_arc_world)
    guided_world = _extract_subpolyline_by_arc_length(
        orbit_world,
        orbit_arc_samples_world,
        start_arc=guided_start_arc_world,
        end_arc=guided_end_arc_world,
        sample_count=guided_sample_count,
    )
    guided_entry_point_world = _sample_polyline_at_arc_lengths(
        orbit_world,
        orbit_arc_samples_world,
        np.array([guided_start_arc_world], dtype=float),
    )[0]
    guided_exit_point_world = _sample_polyline_at_arc_lengths(
        orbit_world,
        orbit_arc_samples_world,
        np.array([guided_end_arc_world], dtype=float),
    )[0]
    entry = {
        "part_id": part["part_id"],
        "curve_fit_path": part["curve_fit_path"],
        "svg_path": part["svg_path"],
        "world_transform": {
            "normalized_center": list(part["normalized_center"]),
            "fit_scale": float(part["fit_scale"]),
        },
        "orbit_duration": float(orbit.time[-1]),
        "orbit_sample_times": orbit_time_targets.tolist(),
        "orbit_points_world": orbit_world.tolist(),
        "orbit_arc_length_samples_world": orbit_arc_samples_world.tolist(),
        "orbit_arc_length_total_world": orbit_total_arc_world,
        "guided_points_world": guided_world.tolist(),
        "guided_entry_point_world": guided_entry_point_world.tolist(),
        "guided_exit_point_world": guided_exit_point_world.tolist(),
        "target_points_world": target_world.tolist(),
        "guided_segment": {
            "start_index": int(model.guided_segment.start_index),
            "end_index": int(model.guided_segment.end_index),
            "start_time": float(model.guided_segment.start_time),
            "end_time": float(model.guided_segment.end_time),
            "mid_time": float(0.5 * (model.guided_segment.start_time + model.guided_segment.end_time)),
            "start_factor": float(model.guided_segment.start_index / total_frames_factor),
            "end_factor": float(model.guided_segment.end_index / total_frames_factor),
            "mid_factor": float(guided_mid_index / total_frames_factor),
            "start_progress": float(guided_start_arc_world / max(orbit_cumulative_world[-1], 1e-12)),
            "end_progress": float(guided_end_arc_world / max(orbit_cumulative_world[-1], 1e-12)),
            "mid_progress": float(guided_mid_arc_world / max(orbit_cumulative_world[-1], 1e-12)),
            "start_arc_length_world": guided_start_arc_world,
            "end_arc_length_world": guided_end_arc_world,
            "mid_arc_length_world": guided_mid_arc_world,
            "max_projection_error": float(model.guided_segment.max_projection_error),
            "mean_projection_error": float(model.guided_segment.mean_projection_error),
        },
        "orbit_object_name": f"{part['part_id']}_orbit",
        "guided_object_name": f"{part['part_id']}_guided",
        "follower_object_name": f"{part['part_id']}_follower",
        "batch_report": batch_report_entry,
    }
    return entry


def export_blender_paths(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    batch_report_path: str | Path | None = None,
    orbit_sample_count: int = 0,
    guided_sample_count: int = 180,
    target_sample_count: int = 180,
    world_config: WorldConfig | None = None,
    limit_parts: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Export world-space path data that Blender can import directly."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    parts = list(manifest["parts"])
    if limit_parts is not None:
        parts = parts[:limit_parts]

    report_lookup = _part_report_lookup(batch_report_path)
    world = world_config or WorldConfig(target_extent=float(manifest["target_extent"]))
    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    viewbox = tuple(float(value) for value in manifest["viewbox"])

    exported_parts: list[dict[str, Any]] = []
    for part in parts:
        fit = load_curve_fit(part["curve_fit_path"])
        model = build_guided_chaotic_field(fit.as_array(), num_curve_samples=fit.num_samples, **resolved_model_kwargs)
        exported_parts.append(
            _export_part_payload(
                part,
                model=model,
                fit=fit,
                viewbox=viewbox,
                world=world,
                orbit_sample_count=orbit_sample_count,
                guided_sample_count=guided_sample_count,
                target_sample_count=target_sample_count,
                batch_report_entry=report_lookup.get(part["part_id"]),
            )
        )

    payload = {
        "manifest_path": str(manifest_path),
        "batch_report_path": None if batch_report_path is None else str(batch_report_path),
        "world": {
            "canvas_extent": float(world.canvas_extent),
            "canvas_z": float(world.canvas_z),
            "depth_scale": float(world.depth_scale),
            "target_extent": float(world.target_extent),
            "plane_axes": list(world.plane_axes),
            "projection_axis": world.projection_axis,
            "local_to_world_depth_scale": float(world.local_to_world_depth_scale()),
        },
        "viewbox": list(viewbox),
        "orbit_sample_count": int(orbit_sample_count),
        "guided_sample_count": int(guided_sample_count),
        "target_sample_count": int(target_sample_count),
        "model_kwargs": resolved_model_kwargs,
        "part_count": len(exported_parts),
        "parts": exported_parts,
    }

    output = Path(output_path) if output_path is not None else Path(manifest["output_dir"]) / "blender_paths.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export world-space guided-attractor paths for Blender.")
    parser.add_argument("manifest", help="Path to dissect_svg manifest.json")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--batch-report", default=None, help="Optional batch_report.json to merge per-part diagnostics")
    parser.add_argument("--orbit-samples", type=int, default=0, help="Number of orbit samples exported per part; use 0 for full-resolution export")
    parser.add_argument("--guided-samples", type=int, default=180, help="Number of guided samples exported per part")
    parser.add_argument("--target-samples", type=int, default=180, help="Number of target samples exported per part")
    parser.add_argument("--canvas-extent", type=float, default=12.0, help="World-space width/height of the square canvas")
    parser.add_argument("--canvas-z", type=float, default=0.0, help="Z height of the projection canvas")
    parser.add_argument("--depth-scale", type=float, default=0.35, help="Relative z-depth scale for the attractors")
    parser.add_argument("--limit-parts", type=int, default=None, help="Optional cap for quick previews")
    parser.add_argument("--total-time", type=float, default=35.0, help="Simulation time passed to core.py")
    parser.add_argument("--transient-time", type=float, default=8.0, help="Transient time discarded before analysis")
    parser.add_argument("--dt", type=float, default=0.005, help="RK4 step size")
    parser.add_argument("--track-height", type=float, default=2.0, help="Guided centerline height")
    parser.add_argument("--return-height", type=float, default=4.0, help="Return centerline height")
    return parser


def main() -> None:
    """CLI entry point for Blender path export."""

    args = _build_parser().parse_args()
    output = export_blender_paths(
        args.manifest,
        output_path=args.output,
        batch_report_path=args.batch_report,
        orbit_sample_count=args.orbit_samples,
        guided_sample_count=args.guided_samples,
        target_sample_count=args.target_samples,
        world_config=WorldConfig(
            canvas_extent=float(args.canvas_extent),
            canvas_z=float(args.canvas_z),
            depth_scale=float(args.depth_scale),
        ),
        limit_parts=args.limit_parts,
        model_kwargs={
            "total_time": float(args.total_time),
            "transient_time": float(args.transient_time),
            "dt": float(args.dt),
            "track_height": float(args.track_height),
            "return_height": float(args.return_height),
        },
    )
    print(json.dumps({"output_path": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
