from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _resolve_enter_frame(enter_frame: int | None, align_frame: int | None) -> int:
    if enter_frame is not None and align_frame is not None and int(enter_frame) != int(align_frame):
        raise ValueError("enter_frame and align_frame must match when both are provided")
    if enter_frame is not None:
        return int(enter_frame)
    if align_frame is not None:
        return int(align_frame)
    return 240


def _build_adaptive_window_schedule(
    payload: dict[str, Any],
    *,
    fps: int,
    enter_frame: int,
    exit_frame: int,
    hold_frames: int,
    reveal_window: tuple[int, int],
    output_path: str | Path | None,
) -> Path:
    if exit_frame <= enter_frame:
        raise ValueError("exit_frame must be greater than enter_frame")

    pre_reveal, post_reveal = reveal_window
    target_window_frames = float(exit_frame - enter_frame)
    parts_schedule: list[dict[str, Any]] = []

    for part in payload["parts"]:
        guided = part["guided_segment"]
        orbit_arc_total = float(part["orbit_arc_length_total_world"])
        guided_start_arc = float(guided["start_arc_length_world"])
        guided_end_arc = float(guided["end_arc_length_world"])
        guided_arc_span = guided_end_arc - guided_start_arc
        if guided_arc_span <= 1e-12:
            raise ValueError(f"guided segment for {part['part_id']} has zero world-space arc length")

        part_frames_per_unit = float(target_window_frames / guided_arc_span)
        raw_start_frame = float(enter_frame - guided_start_arc * part_frames_per_unit)
        raw_end_frame = float(raw_start_frame + orbit_arc_total * part_frames_per_unit)
        travel_frames = float(raw_end_frame - raw_start_frame)
        part_schedule = {
            "part_id": part["part_id"],
            "orbit_object_name": part["orbit_object_name"],
            "guided_object_name": part["guided_object_name"],
            "follower_object_name": part["follower_object_name"],
            "schedule_mode": "adaptive_window",
            "frames_per_unit": part_frames_per_unit,
            "raw_start_frame": raw_start_frame,
            "raw_end_frame": raw_end_frame,
            "raw_enter_target_frame": int(enter_frame),
            "raw_exit_target_frame": int(exit_frame),
            "start_frame": raw_start_frame,
            "end_frame": raw_end_frame,
            "enter_target_frame": int(enter_frame),
            "exit_target_frame": int(exit_frame),
            "align_frame": int(enter_frame),
            "travel_frames": travel_frames,
            "target_window_frames": target_window_frames,
            "orbit_arc_length_total_world": orbit_arc_total,
            "guided_start_arc_length_world": guided_start_arc,
            "guided_end_arc_length_world": guided_end_arc,
            "guided_mid_arc_length_world": float(guided["mid_arc_length_world"]),
            "guided_start_factor": float(guided["start_factor"]),
            "guided_end_factor": float(guided["end_factor"]),
            "guided_mid_factor": float(guided["mid_factor"]),
            "guided_start_progress": float(guided["start_progress"]),
            "guided_end_progress": float(guided["end_progress"]),
            "guided_mid_progress": float(guided["mid_progress"]),
            "guided_start_time": float(guided["start_time"]),
            "guided_end_time": float(guided["end_time"]),
            "guided_mid_time": float(guided["mid_time"]),
            "orbit_duration": float(part["orbit_duration"]),
            "entry_progress": float(guided_start_arc / max(orbit_arc_total, 1e-12)),
            "exit_progress": float(guided_end_arc / max(orbit_arc_total, 1e-12)),
        }
        part_schedule["guided_reveal_start_frame"] = int(enter_frame - pre_reveal)
        part_schedule["guided_reveal_end_frame"] = int(exit_frame + post_reveal)
        part_schedule["hide_before_frame"] = int(math.floor(raw_start_frame) - 1)
        part_schedule["hide_after_frame"] = int(math.ceil(raw_end_frame) + hold_frames)
        parts_schedule.append(part_schedule)

    scene_start = min(item["hide_before_frame"] for item in parts_schedule)
    scene_end = max(item["hide_after_frame"] for item in parts_schedule)
    result = {
        "paths_path": str(payload["paths_path"]) if "paths_path" in payload else None,
        "fps": int(fps),
        "schedule_mode": "adaptive_window",
        "enter_frame_requested": int(enter_frame),
        "exit_frame_requested": int(exit_frame),
        "align_frame_requested": int(enter_frame),
        "align_frame": int(enter_frame),
        "frames_per_unit": None,
        "hold_frames": int(hold_frames),
        "scene_frame_start": int(scene_start),
        "scene_frame_end": int(scene_end),
        "parts": parts_schedule,
    }

    output = Path(output_path) if output_path is not None else Path(payload["paths_path"]).with_name("animation_schedule.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def _build_shared_speed_schedule(
    payload: dict[str, Any],
    *,
    fps: int,
    align_frame: int,
    frames_per_unit: float,
    hold_frames: int,
    reveal_window: tuple[int, int],
    output_path: str | Path | None,
) -> Path:
    pre_reveal, post_reveal = reveal_window
    parts_schedule: list[dict[str, Any]] = []
    raw_start_frames: list[int] = []

    for part in payload["parts"]:
        guided = part["guided_segment"]
        orbit_arc_total = float(part["orbit_arc_length_total_world"])
        guided_start_arc = float(guided["start_arc_length_world"])
        guided_end_arc = float(guided["end_arc_length_world"])
        total_travel_frames = max(1, int(round(orbit_arc_total * frames_per_unit)))
        raw_start_frame = int(round(align_frame - guided_start_arc * frames_per_unit))
        raw_end_frame = raw_start_frame + total_travel_frames
        raw_exit_frame = int(round(align_frame + (guided_end_arc - guided_start_arc) * frames_per_unit))
        part_schedule = {
            "part_id": part["part_id"],
            "orbit_object_name": part["orbit_object_name"],
            "guided_object_name": part["guided_object_name"],
            "follower_object_name": part["follower_object_name"],
            "schedule_mode": "shared_speed",
            "frames_per_unit": float(frames_per_unit),
            "raw_start_frame": int(raw_start_frame),
            "raw_end_frame": int(raw_end_frame),
            "raw_enter_target_frame": int(align_frame),
            "raw_exit_target_frame": int(raw_exit_frame),
            "travel_frames": int(total_travel_frames),
            "orbit_arc_length_total_world": orbit_arc_total,
            "guided_start_arc_length_world": guided_start_arc,
            "guided_end_arc_length_world": guided_end_arc,
            "guided_mid_arc_length_world": float(guided["mid_arc_length_world"]),
            "guided_start_factor": float(guided["start_factor"]),
            "guided_end_factor": float(guided["end_factor"]),
            "guided_mid_factor": float(guided["mid_factor"]),
            "guided_start_progress": float(guided["start_progress"]),
            "guided_end_progress": float(guided["end_progress"]),
            "guided_mid_progress": float(guided["mid_progress"]),
            "guided_start_time": float(guided["start_time"]),
            "guided_end_time": float(guided["end_time"]),
            "guided_mid_time": float(guided["mid_time"]),
            "orbit_duration": float(part["orbit_duration"]),
            "entry_progress": float(guided_start_arc / max(orbit_arc_total, 1e-12)),
            "exit_progress": float(guided_end_arc / max(orbit_arc_total, 1e-12)),
        }
        parts_schedule.append(part_schedule)
        raw_start_frames.append(raw_start_frame)

    shift = -min(raw_start_frames) if raw_start_frames else 0
    effective_align_frame = int(align_frame + shift)
    for item in parts_schedule:
        item["start_frame"] = int(item["raw_start_frame"] + shift)
        item["end_frame"] = int(item["raw_end_frame"] + shift)
        item["enter_target_frame"] = int(item["raw_enter_target_frame"] + shift)
        item["exit_target_frame"] = int(item["raw_exit_target_frame"] + shift)
        item["align_frame"] = effective_align_frame
        item["guided_reveal_start_frame"] = int(item["enter_target_frame"] - pre_reveal)
        item["guided_reveal_end_frame"] = int(item["exit_target_frame"] + post_reveal)
        item["hide_before_frame"] = max(0, item["start_frame"] - 1)
        item["hide_after_frame"] = int(item["end_frame"] + hold_frames)

    scene_end = max(item["hide_after_frame"] for item in parts_schedule)
    result = {
        "paths_path": str(payload["paths_path"]) if "paths_path" in payload else None,
        "fps": int(fps),
        "schedule_mode": "shared_speed",
        "enter_frame_requested": int(align_frame),
        "exit_frame_requested": None,
        "align_frame_requested": int(align_frame),
        "align_frame": int(effective_align_frame),
        "frames_per_unit": float(frames_per_unit),
        "hold_frames": int(hold_frames),
        "scene_frame_start": 0,
        "scene_frame_end": int(scene_end),
        "parts": parts_schedule,
    }

    output = Path(output_path) if output_path is not None else Path(payload["paths_path"]).with_name("animation_schedule.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def build_schedule(
    paths_path: str | Path,
    *,
    output_path: str | Path | None = None,
    fps: int = 24,
    enter_frame: int | None = None,
    exit_frame: int | None = 312,
    align_frame: int | None = None,
    frames_per_unit: float | None = None,
    travel_frames: float | None = None,
    hold_frames: int = 48,
    reveal_window: tuple[int, int] = (24, 24),
) -> Path:
    """Create a global timeline using either adaptive guided-window timing or legacy shared speed."""

    if fps <= 0:
        raise ValueError("fps must be positive")

    payload = json.loads(Path(paths_path).read_text(encoding="utf-8"))
    payload["paths_path"] = str(paths_path)
    resolved_enter_frame = _resolve_enter_frame(enter_frame, align_frame)

    if exit_frame is not None:
        return _build_adaptive_window_schedule(
            payload,
            fps=int(fps),
            enter_frame=int(resolved_enter_frame),
            exit_frame=int(exit_frame),
            hold_frames=int(hold_frames),
            reveal_window=(int(reveal_window[0]), int(reveal_window[1])),
            output_path=output_path,
        )

    resolved_frames_per_unit = float(frames_per_unit if frames_per_unit is not None else (travel_frames if travel_frames is not None else 12.0))
    if resolved_frames_per_unit <= 0:
        raise ValueError("frames_per_unit must be positive")
    return _build_shared_speed_schedule(
        payload,
        fps=int(fps),
        align_frame=int(resolved_enter_frame),
        frames_per_unit=resolved_frames_per_unit,
        hold_frames=int(hold_frames),
        reveal_window=(int(reveal_window[0]), int(reveal_window[1])),
        output_path=output_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a synchronized animation schedule for exported Blender paths.")
    parser.add_argument("paths", help="Path to blender_paths.json")
    parser.add_argument("--output", default=None, help="Output schedule path")
    parser.add_argument("--fps", type=int, default=24, help="Scene frames per second")
    parser.add_argument("--enter-frame", type=int, default=None, help="Frame where all followers should enter their guided target segment")
    parser.add_argument("--exit-frame", type=int, default=312, help="Frame where all followers should leave their guided target segment; set to a negative value to use legacy shared-speed mode")
    parser.add_argument("--align-frame", type=int, default=None, help="Legacy alias for --enter-frame")
    parser.add_argument("--frames-per-unit", type=float, default=None, help="Legacy shared speed measured in frames per world-space arc-length unit; used only when --exit-frame is disabled")
    parser.add_argument("--travel-frames", type=float, default=None, help="Backward-compatible alias for --frames-per-unit in legacy shared-speed mode")
    parser.add_argument("--hold-frames", type=int, default=48, help="Frames to keep objects visible after traversal")
    parser.add_argument("--reveal-window", type=int, nargs=2, default=(24, 24), metavar=("PRE", "POST"), help="Frames before guided entry and after guided exit where helper strokes stay visible")
    return parser


def main() -> None:
    """CLI entry point for schedule generation."""

    args = _build_parser().parse_args()
    output = build_schedule(
        args.paths,
        output_path=args.output,
        fps=int(args.fps),
        enter_frame=None if args.enter_frame is None else int(args.enter_frame),
        exit_frame=None if int(args.exit_frame) < 0 else int(args.exit_frame),
        align_frame=None if args.align_frame is None else int(args.align_frame),
        frames_per_unit=args.frames_per_unit,
        travel_frames=None if args.travel_frames is None else float(args.travel_frames),
        hold_frames=int(args.hold_frames),
        reveal_window=(int(args.reveal_window[0]), int(args.reveal_window[1])),
    )
    print(json.dumps({"output_path": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
