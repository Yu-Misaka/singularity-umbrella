from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_schedule(
    paths_path: str | Path,
    *,
    output_path: str | Path | None = None,
    fps: int = 24,
    align_frame: int = 240,
    travel_frames: int = 360,
    hold_frames: int = 48,
    sync_mode: str = "factor",
    reveal_window: tuple[int, int] = (24, 24),
) -> Path:
    """Create a global timeline so all guided segments align on the same reveal frame."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if travel_frames < 2:
        raise ValueError("travel_frames must be at least 2")
    if sync_mode not in {"factor", "time"}:
        raise ValueError("sync_mode must be either 'factor' or 'time'")

    payload = json.loads(Path(paths_path).read_text(encoding="utf-8"))
    pre_reveal, post_reveal = reveal_window
    parts_schedule: list[dict[str, Any]] = []

    for part in payload["parts"]:
        guided = part["guided_segment"]
        if sync_mode == "factor":
            anchor = float(guided["mid_factor"])
        else:
            orbit_duration = max(float(part["orbit_duration"]), 1e-12)
            anchor = float(guided["mid_time"]) / orbit_duration

        start_frame = int(round(align_frame - anchor * travel_frames))
        end_frame = start_frame + int(travel_frames)
        part_schedule = {
            "part_id": part["part_id"],
            "orbit_object_name": part["orbit_object_name"],
            "guided_object_name": part["guided_object_name"],
            "follower_object_name": part["follower_object_name"],
            "sync_mode": sync_mode,
            "travel_frames": int(travel_frames),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "align_frame": int(align_frame),
            "guided_reveal_start_frame": int(align_frame - pre_reveal),
            "guided_reveal_end_frame": int(align_frame + post_reveal),
            "guided_start_factor": float(guided["start_factor"]),
            "guided_end_factor": float(guided["end_factor"]),
            "guided_mid_factor": float(guided["mid_factor"]),
            "guided_start_time": float(guided["start_time"]),
            "guided_end_time": float(guided["end_time"]),
            "guided_mid_time": float(guided["mid_time"]),
            "orbit_duration": float(part["orbit_duration"]),
            "hide_before_frame": max(1, start_frame - 1),
            "hide_after_frame": int(end_frame + hold_frames),
        }
        parts_schedule.append(part_schedule)

    scene_start = max(1, min(item["hide_before_frame"] for item in parts_schedule))
    scene_end = max(item["hide_after_frame"] for item in parts_schedule)
    result = {
        "paths_path": str(paths_path),
        "fps": int(fps),
        "align_frame": int(align_frame),
        "travel_frames": int(travel_frames),
        "hold_frames": int(hold_frames),
        "sync_mode": sync_mode,
        "scene_frame_start": int(scene_start),
        "scene_frame_end": int(scene_end),
        "parts": parts_schedule,
    }

    output = Path(output_path) if output_path is not None else Path(paths_path).with_name("animation_schedule.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a synchronized animation schedule for exported Blender paths.")
    parser.add_argument("paths", help="Path to blender_paths.json")
    parser.add_argument("--output", default=None, help="Output schedule path")
    parser.add_argument("--fps", type=int, default=24, help="Scene frames per second")
    parser.add_argument("--align-frame", type=int, default=240, help="Frame where all guided midpoints should align")
    parser.add_argument("--travel-frames", type=int, default=360, help="Frames used for one full orbit traversal")
    parser.add_argument("--hold-frames", type=int, default=48, help="Frames to keep objects visible after traversal")
    parser.add_argument("--sync-mode", choices=("factor", "time"), default="factor", help="Align by orbit factor or by local simulation time")
    parser.add_argument("--reveal-window", type=int, nargs=2, default=(24, 24), metavar=("PRE", "POST"), help="Frames before/after align_frame where guided strokes stay highlighted")
    return parser


def main() -> None:
    """CLI entry point for schedule generation."""

    args = _build_parser().parse_args()
    output = build_schedule(
        args.paths,
        output_path=args.output,
        fps=int(args.fps),
        align_frame=int(args.align_frame),
        travel_frames=int(args.travel_frames),
        hold_frames=int(args.hold_frames),
        sync_mode=args.sync_mode,
        reveal_window=(int(args.reveal_window[0]), int(args.reveal_window[1])),
    )
    print(json.dumps({"output_path": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
