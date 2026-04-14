from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


type Array = np.ndarray


@dataclass(slots=True)
class CurveFit:
    """Serializable sampled curve representation shared by the whole pipeline."""

    image_path: str
    image_size: tuple[int, int]
    threshold: int
    bounding_box: tuple[int, int, int, int]
    num_samples: int
    target_extent: float
    samples: list[list[float]]

    def as_array(self) -> Array:
        return np.asarray(self.samples, dtype=float)

    def gamma(self, s: float) -> tuple[float, float]:
        samples = self.as_array()
        clipped = float(np.clip(s, 0.0, 1.0))
        scaled = clipped * (len(samples) - 1)
        lower = int(np.floor(scaled))
        upper = min(lower + 1, len(samples) - 1)
        weight = scaled - lower
        point = (1.0 - weight) * samples[lower] + weight * samples[upper]
        return float(point[0]), float(point[1])

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return output


def load_curve_fit(path: str | Path) -> CurveFit:
    """Load a previously saved `CurveFit` JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CurveFit(**data)


def _grayscale_array(image_path: str | Path) -> Array:
    """Read an image as an 8-bit grayscale array."""

    image = Image.open(image_path).convert("L")
    return np.asarray(image, dtype=np.uint8)


def _otsu_threshold(gray: Array) -> int:
    """Choose a foreground/background split with Otsu's method."""

    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = float(gray.size)
    sum_total = float(np.dot(np.arange(256), histogram))

    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 127
    best_variance = -1.0

    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background <= 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break

        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return int(best_threshold)


def _binary_dilation(mask: Array) -> Array:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    views = []
    for di in range(3):
        for dj in range(3):
            views.append(padded[di : di + mask.shape[0], dj : dj + mask.shape[1]])
    return np.logical_or.reduce(views)


def _binary_erosion(mask: Array) -> Array:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    views = []
    for di in range(3):
        for dj in range(3):
            views.append(padded[di : di + mask.shape[0], dj : dj + mask.shape[1]])
    return np.logical_and.reduce(views)


def _binary_close(mask: Array) -> Array:
    return _binary_erosion(_binary_dilation(mask))


def _largest_component(mask: Array) -> Array:
    """Keep only the largest 8-connected foreground component."""

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_component: list[tuple[int, int]] = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    foreground = np.argwhere(mask)
    for row, col in foreground:
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        component: list[tuple[int, int]] = []

        while stack:
            cr, cc = stack.pop()
            component.append((cr, cc))
            for dr, dc in neighbors:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        if len(component) > len(best_component):
            best_component = component

    if not best_component:
        raise ValueError("no dark curve pixels were found in the image")

    result = np.zeros_like(mask, dtype=bool)
    rows, cols = zip(*best_component, strict=False)
    result[np.array(rows), np.array(cols)] = True
    return result


def _neighbors(mask: Array, row: int, col: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                points.append((nr, nc))
    return points


def _weighted_neighbors(mask: Array, row: int, col: int) -> list[tuple[tuple[int, int], float]]:
    points: list[tuple[tuple[int, int], float]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                points.append(((nr, nc), float(np.hypot(dr, dc))))
    return points


def _zhang_suen_thinning(mask: Array) -> Array:
    skeleton = mask.copy()
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            to_remove: list[tuple[int, int]] = []
            rows, cols = np.nonzero(skeleton)
            for row, col in zip(rows, cols, strict=False):
                if row == 0 or col == 0 or row == skeleton.shape[0] - 1 or col == skeleton.shape[1] - 1:
                    continue
                p2 = skeleton[row - 1, col]
                p3 = skeleton[row - 1, col + 1]
                p4 = skeleton[row, col + 1]
                p5 = skeleton[row + 1, col + 1]
                p6 = skeleton[row + 1, col]
                p7 = skeleton[row + 1, col - 1]
                p8 = skeleton[row, col - 1]
                p9 = skeleton[row - 1, col - 1]
                neighborhood = [p2, p3, p4, p5, p6, p7, p8, p9]
                count = int(sum(neighborhood))
                if count < 2 or count > 6:
                    continue
                transitions = int(
                    (not p2 and p3)
                    + (not p3 and p4)
                    + (not p4 and p5)
                    + (not p5 and p6)
                    + (not p6 and p7)
                    + (not p7 and p8)
                    + (not p8 and p9)
                    + (not p9 and p2)
                )
                if transitions != 1:
                    continue
                if step == 0:
                    if p2 and p4 and p6:
                        continue
                    if p4 and p6 and p8:
                        continue
                else:
                    if p2 and p4 and p8:
                        continue
                    if p2 and p6 and p8:
                        continue
                to_remove.append((int(row), int(col)))
            if to_remove:
                changed = True
                for row, col in to_remove:
                    skeleton[row, col] = False

    return skeleton


def _find_endpoints(mask: Array) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    rows, cols = np.nonzero(mask)
    for row, col in zip(rows, cols, strict=False):
        if len(_neighbors(mask, int(row), int(col))) == 1:
            endpoints.append((int(row), int(col)))
    return endpoints


def _farthest_pixel(mask: Array, start: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int] | None]]:
    """Run Dijkstra on the foreground graph and return the farthest reachable pixel."""

    distances: dict[tuple[int, int], float] = {start: 0.0}
    previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for neighbor, weight in _weighted_neighbors(mask, *node):
            candidate = distance + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    farthest = max(distances, key=distances.get)
    return farthest, distances, previous


def _trace_curve(mask: Array) -> Array:
    """Extract the dominant stroke as the longest path in the foreground graph."""

    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("mask is empty after preprocessing")

    start = (int(rows[0]), int(cols[0]))
    end_a, _, _ = _farthest_pixel(mask, start)
    end_b, _, previous = _farthest_pixel(mask, end_a)

    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = end_b
    while cursor is not None:
        path.append(cursor)
        cursor = previous.get(cursor)

    path.reverse()
    ordered = np.array([(col, row) for row, col in path], dtype=float)
    return ordered


def _resample_polyline(points: Array, num_samples: int) -> Array:
    if len(points) < 2:
        raise ValueError("need at least two points to resample a curve")

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]
    if total <= 1e-12:
        return np.repeat(points[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total, num_samples, dtype=float)
    result = np.empty((num_samples, 2), dtype=float)
    segment = 0

    for index, target in enumerate(targets):
        while segment + 1 < len(cumulative) and cumulative[segment + 1] < target:
            segment += 1
        if segment >= len(lengths):
            result[index] = points[-1]
            continue
        local_start = cumulative[segment]
        local_length = max(lengths[segment], 1e-12)
        weight = (target - local_start) / local_length
        result[index] = (1.0 - weight) * points[segment] + weight * points[segment + 1]

    return result


def _normalize_points(points: Array, image_size: Sequence[int]) -> Array:
    """Normalize image coordinates by image size and flip y to Cartesian orientation."""

    width, height = image_size
    normalized = np.empty_like(points, dtype=float)
    normalized[:, 0] = points[:, 0] / max(width - 1, 1)
    normalized[:, 1] = 1.0 - points[:, 1] / max(height - 1, 1)
    return normalized


def _scale_to_target_extent(points: Array, target_extent: float) -> Array:
    """Center the curve and scale its largest axis span to `target_extent`."""

    if target_extent <= 0:
        raise ValueError("target_extent must be positive")
    centered = np.asarray(points, dtype=float) - np.mean(points, axis=0, keepdims=True)
    span = np.ptp(centered, axis=0)
    current_extent = float(max(span[0], span[1], 1e-12))
    scale = target_extent / current_extent
    return centered * scale


def fit_curve_from_image(
    image_path: str | Path,
    *,
    num_samples: int = 140,
    target_extent: float = 6.0,
    threshold: int | None = None,
) -> CurveFit:
    """Extract a sampled planar curve from a white-background black-line image.

    The output samples are centered and rescaled to a fixed geometric extent so
    they can be passed directly into `core.build_guided_chaotic_field`.
    """

    gray = _grayscale_array(image_path)
    chosen_threshold = _otsu_threshold(gray) if threshold is None else int(threshold)
    mask = gray <= chosen_threshold
    mask = _binary_close(mask)
    mask = _largest_component(mask)
    traced_pixels = _trace_curve(mask)
    if len(traced_pixels) < 2:
        raise ValueError("failed to trace a usable curve from the image")
    resampled_pixels = _resample_polyline(traced_pixels, num_samples)
    normalized_samples = _normalize_points(resampled_pixels, image_size=(gray.shape[1], gray.shape[0]))
    scaled_samples = _scale_to_target_extent(normalized_samples, target_extent)

    rows, cols = np.nonzero(mask)
    bounding_box = (
        int(np.min(cols)),
        int(np.min(rows)),
        int(np.max(cols)),
        int(np.max(rows)),
    )

    return CurveFit(
        image_path=str(image_path),
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
        threshold=chosen_threshold,
        bounding_box=bounding_box,
        num_samples=num_samples,
        target_extent=float(target_extent),
        samples=scaled_samples.tolist(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a normalized curve from a white-background black-line image.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", default="curve_fit.json", help="Output JSON path")
    parser.add_argument("--samples", type=int, default=140, help="Number of normalized samples to output")
    parser.add_argument(
        "--target-extent",
        type=float,
        default=6.0,
        help="Rescale the extracted curve so its centered bounding-box max side equals this value",
    )
    parser.add_argument("--threshold", type=int, default=None, help="Optional manual threshold override")
    return parser


def main() -> None:
    """CLI entry point for image-to-curve preprocessing."""

    args = _build_parser().parse_args()
    fit = fit_curve_from_image(
        args.image,
        num_samples=args.samples,
        target_extent=args.target_extent,
        threshold=args.threshold,
    )
    output_path = fit.save(args.output)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "image_size": fit.image_size,
                "threshold": fit.threshold,
                "bounding_box": fit.bounding_box,
                "num_samples": fit.num_samples,
                "target_extent": fit.target_extent,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
