from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from core import GuidedChaoticField, _interpolate_curve_samples, build_demo_model, build_guided_chaotic_field


type Array = np.ndarray


@dataclass(slots=True)
class ComparisonImage:
    output_path: Path
    image_size: tuple[int, int]
    max_projection_error: float
    mean_projection_error: float


def _polyline_bounds(*polylines: Array) -> tuple[float, float, float, float]:
    valid = [np.asarray(polyline, dtype=float) for polyline in polylines if len(polyline)]
    if not valid:
        raise ValueError("at least one non-empty polyline is required")
    stacked = np.vstack(valid)
    xmin, ymin = np.min(stacked, axis=0)
    xmax, ymax = np.max(stacked, axis=0)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _fit_transform(
    bounds: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    margin: int,
) -> tuple[float, Array]:
    xmin, ymin, xmax, ymax = bounds
    span_x = max(xmax - xmin, 1e-9)
    span_y = max(ymax - ymin, 1e-9)
    usable_w = max(width - 2 * margin, 1)
    usable_h = max(height - 2 * margin, 1)
    scale = min(usable_w / span_x, usable_h / span_y)

    offset_x = 0.5 * (width - scale * (xmin + xmax))
    offset_y = 0.5 * (height + scale * (ymin + ymax))
    transform = np.array([[scale, 0.0], [0.0, -scale]], dtype=float)
    offset = np.array([offset_x, offset_y], dtype=float)
    return scale, np.vstack([transform, offset])


def _map_points(points: Array, affine: Array) -> list[tuple[float, float]]:
    transform = affine[:2]
    offset = affine[2]
    mapped = points @ transform.T + offset
    return [tuple(map(float, point)) for point in mapped]


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Array,
    affine: Array,
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if len(points) < 2:
        return
    draw.line(_map_points(points, affine), fill=fill, width=width, joint="curve")


def _draw_error_links(
    draw: ImageDraw.ImageDraw,
    guided_projection: Array,
    target_projection: Array,
    affine: Array,
    *,
    stride: int = 8,
    fill: tuple[int, int, int] = (215, 94, 77),
) -> None:
    if len(guided_projection) == 0:
        return
    source = _map_points(guided_projection[::stride], affine)
    target = _map_points(target_projection[::stride], affine)
    for start, end in zip(source, target, strict=False):
        draw.line([start, end], fill=fill, width=1)


def _legend(draw: ImageDraw.ImageDraw, *, width: int, model: GuidedChaoticField) -> None:
    segment = model.guided_segment
    lines = [
        ("Orbit projection", (70, 78, 92)),
        ("Target curve", (220, 63, 47)),
        ("Guided segment", (25, 130, 196)),
    ]
    box = (18, 18, min(width - 18, 358), 144)
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255), outline=(210, 214, 220), width=1)
    x0, y0 = box[0] + 14, box[1] + 14
    for index, (label, color) in enumerate(lines):
        y = y0 + 18 * index
        draw.line([(x0, y + 6), (x0 + 26, y + 6)], fill=color, width=3)
        draw.text((x0 + 34, y), label, fill=(30, 36, 43))

    stats_y = y0 + 64
    stats = [
        f"max err: {segment.max_projection_error:.4f}",
        f"mean err: {segment.mean_projection_error:.4f}",
        f"time window: [{segment.start_time:.2f}, {segment.end_time:.2f}]",
    ]
    for index, text in enumerate(stats):
        draw.text((x0, stats_y + 16 * index), text, fill=(55, 65, 81))


def render_comparison_image(
    model: GuidedChaoticField,
    *,
    output_path: str | Path = "comparison.png",
    image_size: tuple[int, int] = (1400, 1000),
    background: tuple[int, int, int] = (248, 249, 251),
) -> ComparisonImage:
    width, height = image_size
    if width < 200 or height < 200:
        raise ValueError("image_size is too small to render a useful comparison")

    output = Path(output_path)
    image = Image.new("RGB", image_size, color=background)
    draw = ImageDraw.Draw(image)

    orbit_projection = model.project(model.orbit.spatial)
    guided_projection = model.guided_projection()
    local_theta = model.orbit.abstract[
        model.guided_segment.start_index : model.guided_segment.end_index + 1,
        0,
    ] % 1.0
    target_projection = _interpolate_curve_samples(
        model.curve_samples,
        np.clip(local_theta / float(model.metadata["track_fraction"]), 0.0, 1.0),
    )

    bounds = _polyline_bounds(orbit_projection, model.curve_samples, guided_projection)
    _, affine = _fit_transform(bounds, width=width, height=height, margin=72)

    _draw_polyline(draw, orbit_projection, affine, fill=(70, 78, 92), width=2)
    _draw_polyline(draw, model.curve_samples, affine, fill=(220, 63, 47), width=4)
    _draw_polyline(draw, guided_projection, affine, fill=(25, 130, 196), width=5)
    _draw_error_links(draw, guided_projection, target_projection, affine)
    _legend(draw, width=width, model=model)

    image.save(output)
    return ComparisonImage(
        output_path=output,
        image_size=image_size,
        max_projection_error=model.guided_segment.max_projection_error,
        mean_projection_error=model.guided_segment.mean_projection_error,
    )


def build_and_render(
    gamma,
    *,
    output_path: str | Path = "comparison.png",
    image_size: tuple[int, int] = (1400, 1000),
    model_kwargs: dict | None = None,
) -> ComparisonImage:
    kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model = build_guided_chaotic_field(gamma, **kwargs)
    return render_comparison_image(model, output_path=output_path, image_size=image_size)


def demo_curve(s: float) -> tuple[float, float]:
    angle = 2.0 * np.pi * s
    radius = 4.0 + 0.6 * np.cos(3.0 * angle)
    return (
        radius * np.cos(angle),
        0.8 * radius * np.sin(angle),
    )


if __name__ == "__main__":
    output = render_comparison_image(build_demo_model(), output_path="comparison.png")
    print(
        {
            "output_path": str(output.output_path),
            "image_size": output.image_size,
            "max_projection_error": round(output.max_projection_error, 6),
            "mean_projection_error": round(output.mean_projection_error, 6),
        }
    )
