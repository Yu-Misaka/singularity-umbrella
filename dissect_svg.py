from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from preprocess_svg import (
    Array,
    _normalize_svg_points,
    _polyline_length,
    extract_svg_polylines,
    fit_curve_from_svg,
)


@dataclass(slots=True)
class PartRecord:
    """Metadata for one accepted SVG subcurve written to disk."""

    part_id: str
    source_subpath_index: int
    split_index: int
    source_length: float
    source_points: int
    source_closed: bool
    source_bbox: tuple[float, float, float, float]
    normalized_center: tuple[float, float]
    fit_scale: float
    svg_path: str
    curve_fit_path: str
    num_samples: int
    target_extent: float


@dataclass(slots=True)
class RejectedPart:
    """Reason why an extracted SVG subcurve was dropped before export."""

    source_subpath_index: int
    split_index: int
    reason: str
    source_length: float
    normalized_extent: float


def _polyline_bounds(points: Array) -> tuple[float, float, float, float]:
    return (
        float(np.min(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 0])),
        float(np.max(points[:, 1])),
    )


def _is_closed(points: Array, tolerance: float = 1e-6) -> bool:
    if len(points) < 3:
        return False
    return bool(np.linalg.norm(points[0] - points[-1]) <= tolerance)


def _sample_along_polyline(points: Array, distances: Array, *, closed: bool) -> Array:
    base = np.asarray(points, dtype=float)
    if closed and _is_closed(base):
        base = base[:-1]
    if len(base) < 2:
        raise ValueError("need at least two points to sample a polyline interval")

    if closed:
        base = np.vstack([base, base[0]])
    deltas = np.diff(base, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-12:
        return np.repeat(base[:1], len(distances), axis=0)

    if closed:
        lookup = np.mod(distances, total_length)
    else:
        lookup = np.clip(distances, 0.0, total_length)

    result = np.empty((len(distances), 2), dtype=float)
    for index, distance in enumerate(lookup):
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = max(0, min(segment, len(lengths) - 1))
        local_start = cumulative[segment]
        local_length = max(lengths[segment], 1e-12)
        weight = (distance - local_start) / local_length
        result[index] = (1.0 - weight) * base[segment] + weight * base[segment + 1]
    return result


def _split_polyline(points: Array, *, max_length: float) -> list[Array]:
    """Split an SVG polyline into shorter curve pieces by arclength."""

    length = _polyline_length(points)
    if max_length <= 0 or length <= max_length:
        return [np.asarray(points, dtype=float)]

    closed = _is_closed(points)
    segment_count = max(2, int(math.ceil(length / max_length)))
    segment_length = length / segment_count
    pieces: list[Array] = []

    if closed:
        sample_count = max(24, int(math.ceil(len(points) / segment_count)) + 1)
        for split_index in range(segment_count):
            start = split_index * segment_length
            stop = start + segment_length
            distances = np.linspace(start, stop, sample_count, dtype=float)
            pieces.append(_sample_along_polyline(points, distances, closed=True))
        return pieces

    sample_count = max(24, int(math.ceil(len(points) / segment_count)))
    for split_index in range(segment_count):
        start = split_index * segment_length
        stop = length if split_index == segment_count - 1 else (split_index + 1) * segment_length
        distances = np.linspace(start, stop, sample_count, dtype=float)
        pieces.append(_sample_along_polyline(points, distances, closed=False))
    return pieces


def _fit_transform_metadata(points: Array, *, viewbox: tuple[float, float, float, float], target_extent: float) -> tuple[Array, Array, float]:
    """Mirror the centering/scaling used by `CurveFit` and expose its inverse map."""

    normalized = _normalize_svg_points(points, viewbox=viewbox)
    center = np.mean(normalized, axis=0)
    centered = normalized - center
    extent = float(max(np.ptp(centered[:, 0]), np.ptp(centered[:, 1]), 1e-12))
    scale = float(target_extent / extent)
    return normalized, center, scale


def _path_data_from_polyline(points: Array) -> str:
    coords = [f"{point[0]:.6f},{point[1]:.6f}" for point in points]
    if not coords:
        raise ValueError("cannot export an empty polyline")
    return "M " + " L ".join(coords)


def _write_part_svg(
    points: Array,
    *,
    output_path: Path,
    viewbox: tuple[float, float, float, float],
) -> None:
    min_x, min_y, width, height = viewbox
    output_path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                (
                    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x:.6f} {min_y:.6f} '
                    f'{width:.6f} {height:.6f}" width="{width:.6f}" height="{height:.6f}">'
                ),
                f'  <path d="{_path_data_from_polyline(points)}" fill="none" stroke="black" stroke-width="1"/>',
                "</svg>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _render_overview(
    accepted_parts: list[tuple[str, Array]],
    *,
    viewbox: tuple[float, float, float, float],
    output_path: Path,
    image_size: tuple[int, int] = (1400, 1400),
) -> None:
    """Render accepted SVG parts in original coordinates with stable part labels."""

    width, height = image_size
    min_x, min_y, vb_width, vb_height = viewbox
    scale = min((width - 80) / max(vb_width, 1e-9), (height - 80) / max(vb_height, 1e-9))
    offset_x = 0.5 * (width - scale * vb_width)
    offset_y = 0.5 * (height - scale * vb_height)

    def map_points(points: Array) -> list[tuple[float, float]]:
        mapped_x = offset_x + scale * (points[:, 0] - min_x)
        mapped_y = height - (offset_y + scale * (points[:, 1] - min_y))
        return list(zip(mapped_x.tolist(), mapped_y.tolist(), strict=False))

    image = Image.new("RGB", image_size, (250, 251, 253))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 20, width - 20, height - 20), radius=18, outline=(220, 224, 231), width=2)

    palette = [
        (208, 76, 64),
        (47, 119, 191),
        (36, 158, 117),
        (199, 131, 30),
        (128, 88, 183),
        (198, 60, 140),
    ]
    for index, (part_id, points) in enumerate(accepted_parts):
        color = palette[index % len(palette)]
        mapped = map_points(points)
        if len(mapped) >= 2:
            draw.line(mapped, fill=color, width=3)
        label_point = mapped[len(mapped) // 2]
        draw.rounded_rectangle(
            (label_point[0] - 16, label_point[1] - 10, label_point[0] + 16, label_point[1] + 10),
            radius=8,
            fill=(255, 255, 255),
            outline=color,
            width=1,
        )
        draw.text((label_point[0] - 12, label_point[1] - 7), part_id.split("_")[-1], fill=(30, 35, 42))

    image.save(output_path)


def dissect_svg_to_directory(
    svg_path: str | Path,
    *,
    output_dir: str | Path,
    num_samples: int = 140,
    target_extent: float = 6.0,
    min_svg_length: float = 15.0,
    min_normalized_extent: float = 0.02,
    max_svg_length: float = 140.0,
) -> dict[str, object]:
    """Split a complex SVG into manageable curve parts plus per-part `CurveFit` files."""

    svg_path = Path(svg_path)
    output_dir = Path(output_dir)
    parts_dir = output_dir / "parts"
    fits_dir = output_dir / "curve_fits"
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    fits_dir.mkdir(parents=True, exist_ok=True)

    viewbox, subpaths = extract_svg_polylines(svg_path)
    accepted_records: list[PartRecord] = []
    accepted_overview: list[tuple[str, Array]] = []
    rejected_records: list[RejectedPart] = []

    part_counter = 0
    for subpath_index, subpath in enumerate(subpaths):
        for split_index, piece in enumerate(_split_polyline(subpath, max_length=max_svg_length)):
            length = _polyline_length(piece)
            normalized, center, scale = _fit_transform_metadata(piece, viewbox=viewbox, target_extent=target_extent)
            normalized_extent = float(max(np.ptp(normalized[:, 0]), np.ptp(normalized[:, 1]), 0.0))
            if length < min_svg_length:
                rejected_records.append(
                    RejectedPart(
                        source_subpath_index=subpath_index,
                        split_index=split_index,
                        reason="too_short",
                        source_length=float(length),
                        normalized_extent=normalized_extent,
                    )
                )
                continue
            if normalized_extent < min_normalized_extent:
                rejected_records.append(
                    RejectedPart(
                        source_subpath_index=subpath_index,
                        split_index=split_index,
                        reason="too_small",
                        source_length=float(length),
                        normalized_extent=normalized_extent,
                    )
                )
                continue

            part_id = f"part_{part_counter:03d}"
            part_svg_path = parts_dir / f"{part_id}.svg"
            curve_fit_path = fits_dir / f"{part_id}.json"
            _write_part_svg(piece, output_path=part_svg_path, viewbox=viewbox)
            fit = fit_curve_from_svg(part_svg_path, num_samples=num_samples, target_extent=target_extent)
            fit.save(curve_fit_path)

            record = PartRecord(
                part_id=part_id,
                source_subpath_index=subpath_index,
                split_index=split_index,
                source_length=float(length),
                source_points=int(len(piece)),
                source_closed=_is_closed(piece),
                source_bbox=_polyline_bounds(piece),
                normalized_center=(float(center[0]), float(center[1])),
                fit_scale=float(scale),
                svg_path=str(part_svg_path),
                curve_fit_path=str(curve_fit_path),
                num_samples=num_samples,
                target_extent=float(target_extent),
            )
            accepted_records.append(record)
            accepted_overview.append((part_id, piece))
            part_counter += 1

    overview_path = output_dir / "overview.png"
    _render_overview(accepted_overview, viewbox=viewbox, output_path=overview_path)

    manifest = {
        "svg_path": str(svg_path),
        "output_dir": str(output_dir),
        "viewbox": [float(value) for value in viewbox],
        "image_size": [int(round(viewbox[2])), int(round(viewbox[3]))],
        "num_samples": int(num_samples),
        "target_extent": float(target_extent),
        "min_svg_length": float(min_svg_length),
        "min_normalized_extent": float(min_normalized_extent),
        "max_svg_length": float(max_svg_length),
        "source_subpath_count": int(len(subpaths)),
        "accepted_part_count": int(len(accepted_records)),
        "rejected_part_count": int(len(rejected_records)),
        "overview_path": str(overview_path),
        "parts": [asdict(record) for record in accepted_records],
        "rejected": [asdict(record) for record in rejected_records],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dissect a complex SVG into filtered curve parts.")
    parser.add_argument("svg", help="Input SVG path")
    parser.add_argument("--output-dir", default="experiment_outputs/canary_1", help="Directory for the dissected outputs")
    parser.add_argument("--samples", type=int, default=140, help="Number of samples per exported CurveFit")
    parser.add_argument("--target-extent", type=float, default=6.0, help="Local geometric extent used by downstream core fitting")
    parser.add_argument("--min-svg-length", type=float, default=15.0, help="Reject parts shorter than this SVG arclength")
    parser.add_argument("--min-normalized-extent", type=float, default=0.02, help="Reject parts whose normalized bbox max side is smaller than this")
    parser.add_argument("--max-svg-length", type=float, default=140.0, help="Split very long subpaths into pieces no longer than this arclength")
    return parser


def main() -> None:
    """CLI entry point for SVG dissection."""

    args = _build_parser().parse_args()
    manifest = dissect_svg_to_directory(
        args.svg,
        output_dir=args.output_dir,
        num_samples=args.samples,
        target_extent=args.target_extent,
        min_svg_length=args.min_svg_length,
        min_normalized_extent=args.min_normalized_extent,
        max_svg_length=args.max_svg_length,
    )
    print(
        json.dumps(
            {
                "manifest_path": manifest["manifest_path"],
                "accepted_part_count": manifest["accepted_part_count"],
                "rejected_part_count": manifest["rejected_part_count"],
                "overview_path": manifest["overview_path"],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
