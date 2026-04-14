from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from core import GuidedChaoticField, build_guided_chaotic_field
from preprocess import CurveFit, load_curve_fit


type Array = np.ndarray


@dataclass(slots=True)
class BatchComparisonResult:
    """Summary of one batch-composed SVG experiment."""

    manifest_path: Path
    output_path: Path
    report_path: Path
    image_size: tuple[int, int]
    accepted_part_count: int
    mean_projection_error: float
    max_projection_error: float


def _viewbox_affine(
    viewbox: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
    margin: int,
) -> tuple[float, float, float]:
    min_x, min_y, width, height = viewbox
    canvas_w, canvas_h = image_size
    scale = min((canvas_w - 2 * margin) / max(width, 1e-9), (canvas_h - 2 * margin) / max(height, 1e-9))
    offset_x = 0.5 * (canvas_w - scale * width) - scale * min_x
    offset_y = 0.5 * (canvas_h - scale * height) - scale * min_y
    return float(scale), float(offset_x), float(offset_y)


def _svg_to_canvas(points: Array, *, viewbox: tuple[float, float, float, float], image_size: tuple[int, int], margin: int) -> list[tuple[float, float]]:
    scale, offset_x, offset_y = _viewbox_affine(viewbox, image_size=image_size, margin=margin)
    height = image_size[1]
    mapped_x = offset_x + scale * points[:, 0]
    mapped_y = height - (offset_y + scale * points[:, 1])
    return list(zip(mapped_x.tolist(), mapped_y.tolist(), strict=False))


def _local_to_svg(
    points: Array,
    *,
    normalized_center: tuple[float, float],
    fit_scale: float,
    viewbox: tuple[float, float, float, float],
) -> Array:
    """Invert the per-part centering/scaling and map back into SVG coordinates."""

    if fit_scale <= 0:
        raise ValueError("fit_scale must be positive")
    center = np.asarray(normalized_center, dtype=float)
    normalized = np.asarray(points, dtype=float) / float(fit_scale) + center
    min_x, min_y, width, height = viewbox
    restored = np.empty_like(normalized)
    restored[:, 0] = min_x + normalized[:, 0] * width
    restored[:, 1] = min_y + (1.0 - normalized[:, 1]) * height
    return restored


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Array,
    *,
    viewbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    margin: int,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    if len(points) < 2:
        return
    draw.line(
        _svg_to_canvas(points, viewbox=viewbox, image_size=image_size, margin=margin),
        fill=fill,
        width=width,
        joint="curve",
    )


def _compose_layers(
    *,
    image_size: tuple[int, int],
    orbit_paths: list[Array],
    guided_paths: list[Array],
    target_paths: list[Array],
    viewbox: tuple[float, float, float, float],
    margin: int,
) -> Image.Image:
    """Render target curves, faint attractors, and highlighted guided segments."""

    background = Image.new("RGBA", image_size, (248, 249, 251, 255))
    orbit_layer = Image.new("RGBA", image_size, (255, 255, 255, 0))
    target_layer = Image.new("RGBA", image_size, (255, 255, 255, 0))
    guided_layer = Image.new("RGBA", image_size, (255, 255, 255, 0))

    orbit_draw = ImageDraw.Draw(orbit_layer, "RGBA")
    target_draw = ImageDraw.Draw(target_layer, "RGBA")
    guided_draw = ImageDraw.Draw(guided_layer, "RGBA")

    for path in orbit_paths:
        _draw_polyline(
            orbit_draw,
            path,
            viewbox=viewbox,
            image_size=image_size,
            margin=margin,
            fill=(122, 168, 214, 34),
            width=1,
        )
    for path in target_paths:
        _draw_polyline(
            target_draw,
            path,
            viewbox=viewbox,
            image_size=image_size,
            margin=margin,
            fill=(45, 52, 61, 210),
            width=2,
        )
    for path in guided_paths:
        _draw_polyline(
            guided_draw,
            path,
            viewbox=viewbox,
            image_size=image_size,
            margin=margin,
            fill=(29, 111, 183, 228),
            width=3,
        )

    composite = Image.alpha_composite(background, orbit_layer)
    composite = Image.alpha_composite(composite, target_layer)
    composite = Image.alpha_composite(composite, guided_layer)

    draw = ImageDraw.Draw(composite)
    draw.rounded_rectangle((20, 20, min(image_size[0] - 20, 360), 140), radius=12, fill=(255, 255, 255, 228), outline=(214, 219, 226), width=1)
    legend = [
        ("Target curves", (45, 52, 61)),
        ("Guided segments", (29, 111, 183)),
        ("Full attractors", (122, 168, 214)),
    ]
    for idx, (label, color) in enumerate(legend):
        y = 38 + 24 * idx
        draw.line([(38, y), (66, y)], fill=color, width=3)
        draw.text((78, y - 8), label, fill=(33, 38, 45))
    return composite.convert("RGB")


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_model_for_part(part: dict[str, Any], *, model_kwargs: dict[str, Any]) -> tuple[CurveFit, GuidedChaoticField]:
    fit = load_curve_fit(part["curve_fit_path"])
    model = build_guided_chaotic_field(fit.as_array(), num_curve_samples=fit.num_samples, **model_kwargs)
    return fit, model


def compose_manifest(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    report_path: str | Path | None = None,
    image_size: tuple[int, int] = (1800, 1800),
    margin: int = 64,
    limit_parts: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> BatchComparisonResult:
    """Run the guided-field pipeline for each dissected SVG part and merge the overlays."""

    manifest = _load_manifest(manifest_path)
    viewbox = tuple(float(value) for value in manifest["viewbox"])
    parts = list(manifest["parts"])
    if limit_parts is not None:
        parts = parts[:limit_parts]

    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    orbit_paths: list[Array] = []
    guided_paths: list[Array] = []
    target_paths: list[Array] = []
    part_reports: list[dict[str, Any]] = []

    for part in parts:
        fit, model = _build_model_for_part(part, model_kwargs=resolved_model_kwargs)
        orbit_svg = _local_to_svg(
            model.project(model.orbit.spatial),
            normalized_center=tuple(part["normalized_center"]),
            fit_scale=float(part["fit_scale"]),
            viewbox=viewbox,
        )
        guided_svg = _local_to_svg(
            model.guided_projection(),
            normalized_center=tuple(part["normalized_center"]),
            fit_scale=float(part["fit_scale"]),
            viewbox=viewbox,
        )
        target_svg = _local_to_svg(
            fit.as_array(),
            normalized_center=tuple(part["normalized_center"]),
            fit_scale=float(part["fit_scale"]),
            viewbox=viewbox,
        )
        orbit_paths.append(orbit_svg)
        guided_paths.append(guided_svg)
        target_paths.append(target_svg)
        part_reports.append(
            {
                "part_id": part["part_id"],
                "curve_fit_path": part["curve_fit_path"],
                "svg_path": part["svg_path"],
                "max_projection_error": float(model.guided_segment.max_projection_error),
                "mean_projection_error": float(model.guided_segment.mean_projection_error),
                "start_time": float(model.guided_segment.start_time),
                "end_time": float(model.guided_segment.end_time),
                "separation_peak_ratio": float(model.metadata["separation_peak_ratio"]),
                "separation_final_ratio": float(model.metadata["separation_final_ratio"]),
            }
        )

    output = Path(output_path) if output_path is not None else Path(manifest["output_dir"]) / "batch_comparison.png"
    report = Path(report_path) if report_path is not None else Path(manifest["output_dir"]) / "batch_report.json"

    composite = _compose_layers(
        image_size=image_size,
        orbit_paths=orbit_paths,
        guided_paths=guided_paths,
        target_paths=target_paths,
        viewbox=viewbox,
        margin=margin,
    )
    composite.save(output)

    errors = [item["mean_projection_error"] for item in part_reports]
    max_errors = [item["max_projection_error"] for item in part_reports]
    report_payload = {
        "manifest_path": str(Path(manifest_path)),
        "output_path": str(output),
        "image_size": list(image_size),
        "accepted_part_count": len(part_reports),
        "mean_projection_error": float(np.mean(errors)) if errors else 0.0,
        "max_projection_error": float(np.max(max_errors)) if max_errors else 0.0,
        "parts": part_reports,
        "model_kwargs": resolved_model_kwargs,
    }
    report.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    return BatchComparisonResult(
        manifest_path=Path(manifest_path),
        output_path=output,
        report_path=report,
        image_size=image_size,
        accepted_part_count=len(part_reports),
        mean_projection_error=float(report_payload["mean_projection_error"]),
        max_projection_error=float(report_payload["max_projection_error"]),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run core.py over dissected SVG parts and composite the overlays.")
    parser.add_argument("manifest", help="Path to the dissect_svg manifest.json")
    parser.add_argument("--output", default=None, help="Composite output image path")
    parser.add_argument("--report", default=None, help="JSON report path")
    parser.add_argument("--image-size", type=int, nargs=2, default=(1800, 1800), metavar=("WIDTH", "HEIGHT"), help="Composite image size")
    parser.add_argument("--margin", type=int, default=64, help="Canvas margin in pixels")
    parser.add_argument("--limit-parts", type=int, default=None, help="Optional cap for quick canary runs")
    parser.add_argument("--total-time", type=float, default=35.0, help="Per-part simulation time")
    parser.add_argument("--transient-time", type=float, default=8.0, help="Per-part transient discard time")
    parser.add_argument("--dt", type=float, default=0.005, help="Per-part RK4 step size")
    parser.add_argument("--track-height", type=float, default=2.0, help="Centerline height on the guided portion")
    parser.add_argument("--return-height", type=float, default=4.0, help="Centerline height on the return portion")
    return parser


def main() -> None:
    """CLI entry point for batch composition."""

    args = _build_parser().parse_args()
    result = compose_manifest(
        args.manifest,
        output_path=args.output,
        report_path=args.report,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        margin=args.margin,
        limit_parts=args.limit_parts,
        model_kwargs={
            "total_time": float(args.total_time),
            "transient_time": float(args.transient_time),
            "dt": float(args.dt),
            "track_height": float(args.track_height),
            "return_height": float(args.return_height),
        },
    )
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "report_path": str(result.report_path),
                "accepted_part_count": result.accepted_part_count,
                "mean_projection_error": round(result.mean_projection_error, 6),
                "max_projection_error": round(result.max_projection_error, 6),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
