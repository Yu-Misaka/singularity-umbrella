from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from preprocess import CurveFit, _resample_polyline, _scale_to_target_extent


type Array = np.ndarray


COMMAND_RE = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_length(text: str | None, fallback: float) -> float:
    if text is None:
        return fallback
    match = re.match(r"\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)", text)
    if match is None:
        return fallback
    return float(match.group(1))


def _resolve_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    """Resolve the SVG coordinate box used for normalization."""

    viewbox = root.get("viewBox")
    if viewbox:
        numbers = [float(part) for part in re.split(r"[,\s]+", viewbox.strip()) if part]
        if len(numbers) == 4:
            return numbers[0], numbers[1], numbers[2], numbers[3]
    width = _parse_length(root.get("width"), 1.0)
    height = _parse_length(root.get("height"), 1.0)
    return 0.0, 0.0, width, height


def _sample_cubic(p0: Array, p1: Array, p2: Array, p3: Array, segments: int = 32) -> Array:
    """Sample a cubic Bezier segment as a polyline."""

    t = np.linspace(0.0, 1.0, segments + 1, dtype=float)
    omt = 1.0 - t
    curve = (
        (omt**3)[:, None] * p0
        + (3.0 * omt * omt * t)[:, None] * p1
        + (3.0 * omt * t * t)[:, None] * p2
        + (t**3)[:, None] * p3
    )
    return curve


def _sample_quadratic(p0: Array, p1: Array, p2: Array, segments: int = 24) -> Array:
    """Sample a quadratic Bezier segment as a polyline."""

    t = np.linspace(0.0, 1.0, segments + 1, dtype=float)
    omt = 1.0 - t
    curve = (omt * omt)[:, None] * p0 + (2.0 * omt * t)[:, None] * p1 + (t * t)[:, None] * p2
    return curve


def _tokenize_path_data(path_data: str) -> list[str]:
    tokens = COMMAND_RE.findall(path_data)
    if not tokens:
        raise ValueError("path data is empty")
    return tokens


def _is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def _flatten_path(path_data: str) -> list[Array]:
    """Convert SVG path data into one or more sampled polylines.

    Supported commands are `M/L/H/V/C/S/Q/T/Z` in both absolute and relative
    forms. Elliptic arc commands are intentionally left unsupported for now.
    """

    tokens = _tokenize_path_data(path_data)
    index = 0
    command = ""
    current = np.array([0.0, 0.0], dtype=float)
    subpath_start = current.copy()
    last_cubic_control: Array | None = None
    last_quadratic_control: Array | None = None
    subpaths: list[list[Array]] = []
    points: list[Array] = []

    def read_numbers(count: int) -> list[float]:
        nonlocal index
        if index + count > len(tokens):
            raise ValueError("unexpected end of path data")
        values = [float(tokens[index + offset]) for offset in range(count)]
        index += count
        return values

    def append_point(point: Array) -> None:
        if not points or np.linalg.norm(points[-1] - point) > 1e-12:
            points.append(point.copy())

    while index < len(tokens):
        token = tokens[index]
        if _is_command(token):
            command = token
            index += 1
        elif not command:
            raise ValueError("path data must begin with a command")

        absolute = command.isupper()
        op = command.upper()

        if op == "M":
            x, y = read_numbers(2)
            target = np.array([x, y], dtype=float)
            if not absolute:
                target = current + target
            if points:
                subpaths.append(points)
                points = []
            current = target
            subpath_start = target.copy()
            append_point(current)
            last_cubic_control = None
            last_quadratic_control = None
            command = "L" if absolute else "l"
            continue

        if op == "Z":
            append_point(subpath_start)
            current = subpath_start.copy()
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if op == "L":
            x, y = read_numbers(2)
            target = np.array([x, y], dtype=float)
            if not absolute:
                target = current + target
            current = target
            append_point(current)
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if op == "H":
            (x,) = read_numbers(1)
            target = np.array([x, current[1]], dtype=float) if absolute else np.array([current[0] + x, current[1]], dtype=float)
            current = target
            append_point(current)
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if op == "V":
            (y,) = read_numbers(1)
            target = np.array([current[0], y], dtype=float) if absolute else np.array([current[0], current[1] + y], dtype=float)
            current = target
            append_point(current)
            last_cubic_control = None
            last_quadratic_control = None
            continue

        if op == "C":
            values = read_numbers(6)
            controls = np.array(values, dtype=float).reshape(3, 2)
            if not absolute:
                controls = controls + current
            curve = _sample_cubic(current, controls[0], controls[1], controls[2])
            for point in curve[1:]:
                append_point(point)
            current = controls[2]
            last_cubic_control = controls[1].copy()
            last_quadratic_control = None
            continue

        if op == "S":
            values = read_numbers(4)
            controls = np.array(values, dtype=float).reshape(2, 2)
            if not absolute:
                controls = controls + current
            reflected = current.copy() if last_cubic_control is None else current + (current - last_cubic_control)
            curve = _sample_cubic(current, reflected, controls[0], controls[1])
            for point in curve[1:]:
                append_point(point)
            current = controls[1]
            last_cubic_control = controls[0].copy()
            last_quadratic_control = None
            continue

        if op == "Q":
            values = read_numbers(4)
            controls = np.array(values, dtype=float).reshape(2, 2)
            if not absolute:
                controls = controls + current
            curve = _sample_quadratic(current, controls[0], controls[1])
            for point in curve[1:]:
                append_point(point)
            current = controls[1]
            last_quadratic_control = controls[0].copy()
            last_cubic_control = None
            continue

        if op == "T":
            values = read_numbers(2)
            target = np.array(values, dtype=float)
            if not absolute:
                target = current + target
            reflected = current.copy() if last_quadratic_control is None else current + (current - last_quadratic_control)
            curve = _sample_quadratic(current, reflected, target)
            for point in curve[1:]:
                append_point(point)
            current = target
            last_quadratic_control = reflected.copy()
            last_cubic_control = None
            continue

        if op == "A":
            raise ValueError("elliptical arc commands are not supported yet in preprocess_svg.py")

        raise ValueError(f"unsupported SVG path command: {command}")

    if points:
        subpaths.append(points)

    flattened = [np.asarray(subpath, dtype=float) for subpath in subpaths if len(subpath) >= 2]
    if not flattened:
        raise ValueError("no usable polyline could be extracted from SVG path data")
    return flattened


def _polyline_length(points: Array) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def _normalize_svg_points(points: Array, *, viewbox: tuple[float, float, float, float]) -> Array:
    """Normalize SVG coordinates by viewBox and flip y to Cartesian orientation."""

    min_x, min_y, width, height = viewbox
    if width <= 0 or height <= 0:
        raise ValueError("SVG viewBox must have positive width and height")
    normalized = np.empty_like(points, dtype=float)
    normalized[:, 0] = (points[:, 0] - min_x) / width
    normalized[:, 1] = 1.0 - (points[:, 1] - min_y) / height
    return normalized


def extract_svg_polylines(svg_path: str | Path) -> tuple[tuple[float, float, float, float], list[Array]]:
    """Load an SVG and flatten all `<path>` geometry into sampled polylines."""

    tree = ET.parse(svg_path)
    root = tree.getroot()
    viewbox = _resolve_viewbox(root)

    path_polylines: list[Array] = []
    for element in root.iter():
        if _strip_namespace(element.tag) != "path":
            continue
        path_data = element.get("d")
        if not path_data:
            continue
        path_polylines.extend(_flatten_path(path_data))

    if not path_polylines:
        raise ValueError("no <path> elements with geometry were found in the SVG")
    return viewbox, path_polylines


def curve_fit_from_polyline(
    points: Array,
    *,
    svg_path: str | Path,
    viewbox: tuple[float, float, float, float],
    num_samples: int = 140,
    target_extent: float = 6.0,
) -> CurveFit:
    """Convert one SVG polyline into the shared centered/scaled `CurveFit` format."""

    if len(points) < 2:
        raise ValueError("need at least two SVG points to build a curve fit")

    resampled = _resample_polyline(np.asarray(points, dtype=float), num_samples)
    normalized = _normalize_svg_points(resampled, viewbox=viewbox)
    scaled = _scale_to_target_extent(normalized, target_extent)

    bounds = (
        int(np.floor(np.min(points[:, 0]))),
        int(np.floor(np.min(points[:, 1]))),
        int(np.ceil(np.max(points[:, 0]))),
        int(np.ceil(np.max(points[:, 1]))),
    )
    image_size = (int(round(viewbox[2])), int(round(viewbox[3])))
    return CurveFit(
        image_path=str(svg_path),
        image_size=image_size,
        threshold=-1,
        bounding_box=bounds,
        num_samples=num_samples,
        target_extent=float(target_extent),
        samples=scaled.tolist(),
    )


def fit_curve_from_svg(
    svg_path: str | Path,
    *,
    num_samples: int = 140,
    target_extent: float = 6.0,
) -> CurveFit:
    """Extract the longest SVG path as a centered, scaled `CurveFit`."""

    viewbox, path_polylines = extract_svg_polylines(svg_path)
    selected = max(path_polylines, key=_polyline_length)
    return curve_fit_from_polyline(
        selected,
        svg_path=svg_path,
        viewbox=viewbox,
        num_samples=num_samples,
        target_extent=target_extent,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a normalized curve from an SVG path.")
    parser.add_argument("svg", help="Input SVG path")
    parser.add_argument("--output", default="curve_fit_svg.json", help="Output JSON path")
    parser.add_argument("--samples", type=int, default=140, help="Number of output samples")
    parser.add_argument(
        "--target-extent",
        type=float,
        default=6.0,
        help="Rescale the extracted curve so its centered bounding-box max side equals this value",
    )
    return parser


def main() -> None:
    """CLI entry point for SVG-to-curve preprocessing."""

    args = _build_parser().parse_args()
    fit = fit_curve_from_svg(args.svg, num_samples=args.samples, target_extent=args.target_extent)
    output_path = fit.save(args.output)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "image_size": fit.image_size,
                "bounding_box": fit.bounding_box,
                "num_samples": fit.num_samples,
                "target_extent": fit.target_extent,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
