"""Build attractor-friendly target densities from ordinary images.

The optimiser in ``main.py`` works best when the target looks like a bounded
probability density rather than a raw line drawing or a photographic luminance
field. This helper turns an input image into a hybrid density map made from:

1. a coarse silhouette term,
2. a band-limited edge term,
3. a soft shading term.

The output follows the convention expected by ``main.py``:
dark ink = high desired density.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class PrepConfig:
    input_path: str
    out_path: str
    preview_path: str | None
    size: int
    foreground: str
    silhouette_weight: float
    edge_weight: float
    shading_weight: float
    edge_blur: float
    shading_blur: float
    final_blur: float
    gamma: float
    threshold_bias: float
    crop: bool
    margin: float


def load_grayscale(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def infer_foreground_polarity(gray: np.ndarray) -> str:
    h, w = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    center = gray[y0:y1, x0:x1]
    return "dark" if float(center.mean()) < float(border.mean()) else "light"


def normalise01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn + 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


def otsu_threshold(values01: np.ndarray) -> float:
    vals = np.clip((values01.ravel() * 255.0).round().astype(np.uint8), 0, 255)
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    score = np.full(256, -1.0, dtype=np.float64)
    valid = denom > 1e-12
    score[valid] = (mu_t * omega[valid] - mu[valid]) ** 2 / denom[valid]
    return float(np.argmax(score) / 255.0)


def blur_array(arr01: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return arr01.astype(np.float32, copy=True)
    img = Image.fromarray((255.0 * np.clip(arr01, 0.0, 1.0)).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def center_crop_from_mask(gray: np.ndarray, density01: np.ndarray, margin: float) -> np.ndarray:
    ys, xs = np.where(density01 > 0.08)
    if ys.size == 0:
        return gray
    h, w = gray.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(1, int(round((y1 - y0) * margin)))
    pad_x = max(1, int(round((x1 - x0) * margin)))
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    cropped = gray[y0:y1, x0:x1]
    side = max(cropped.shape)
    canvas = np.full((side, side), 255.0, dtype=np.float32)
    oy = (side - cropped.shape[0]) // 2
    ox = (side - cropped.shape[1]) // 2
    canvas[oy:oy + cropped.shape[0], ox:ox + cropped.shape[1]] = cropped
    return canvas


def build_density(gray: np.ndarray, cfg: PrepConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray01 = normalise01(gray)
    polarity = cfg.foreground
    if polarity == "auto":
        polarity = infer_foreground_polarity(gray)
    base = 1.0 - gray01 if polarity == "dark" else gray01
    base = normalise01(base)

    smooth_for_mask = blur_array(base, 1.2)
    thresh = np.clip(otsu_threshold(smooth_for_mask) + cfg.threshold_bias, 0.05, 0.95)
    silhouette = (smooth_for_mask >= thresh).astype(np.float32)
    silhouette = blur_array(silhouette, 1.0)

    gx = np.zeros_like(base)
    gy = np.zeros_like(base)
    gx[:, 1:-1] = 0.5 * (base[:, 2:] - base[:, :-2])
    gy[1:-1, :] = 0.5 * (base[2:, :] - base[:-2, :])
    edge = normalise01(np.hypot(gx, gy))
    edge = blur_array(edge, cfg.edge_blur)

    shading = blur_array(base, cfg.shading_blur)
    shading *= np.clip(0.35 + 0.65 * silhouette, 0.0, 1.0)

    density = (
        cfg.silhouette_weight * silhouette
        + cfg.edge_weight * edge
        + cfg.shading_weight * shading
    )
    density = blur_array(normalise01(density), cfg.final_blur)
    density = np.power(np.clip(density, 0.0, 1.0), cfg.gamma)
    density = normalise01(density)
    return density, silhouette, edge, shading


def save_density(path: str, density01: np.ndarray) -> None:
    out = (255.0 * (1.0 - np.clip(density01, 0.0, 1.0))).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)


def save_preview(path: str, gray01: np.ndarray, density01: np.ndarray) -> None:
    left = (255.0 * np.clip(gray01, 0.0, 1.0)).astype(np.uint8)
    right = (255.0 * (1.0 - np.clip(density01, 0.0, 1.0))).astype(np.uint8)
    preview = np.concatenate([left, right], axis=1)
    Image.fromarray(preview, mode="L").save(path)


def parse_args() -> PrepConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Source image path.")
    p.add_argument("--out", required=True, help="Output grayscale density path.")
    p.add_argument("--preview", default=None, help="Optional side-by-side preview path.")
    p.add_argument("--size", type=int, default=128, help="Square working resolution.")
    p.add_argument("--foreground", choices=["auto", "dark", "light"], default="auto")
    p.add_argument("--silhouette-weight", type=float, default=0.60)
    p.add_argument("--edge-weight", type=float, default=1.35)
    p.add_argument("--shading-weight", type=float, default=0.55)
    p.add_argument("--edge-blur", type=float, default=1.0)
    p.add_argument("--shading-blur", type=float, default=4.0)
    p.add_argument("--final-blur", type=float, default=1.2)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--threshold-bias", type=float, default=-0.08)
    p.add_argument("--crop", action="store_true", help="Crop to the foreground support.")
    p.add_argument("--margin", type=float, default=0.08)
    args = p.parse_args()
    return PrepConfig(
        input_path=args.input,
        out_path=args.out,
        preview_path=args.preview,
        size=args.size,
        foreground=args.foreground,
        silhouette_weight=args.silhouette_weight,
        edge_weight=args.edge_weight,
        shading_weight=args.shading_weight,
        edge_blur=args.edge_blur,
        shading_blur=args.shading_blur,
        final_blur=args.final_blur,
        gamma=args.gamma,
        threshold_bias=args.threshold_bias,
        crop=args.crop,
        margin=args.margin,
    )


def main() -> None:
    cfg = parse_args()
    gray = load_grayscale(cfg.input_path, cfg.size)

    if cfg.crop:
        gray01 = normalise01(gray)
        polarity = cfg.foreground
        if polarity == "auto":
            polarity = infer_foreground_polarity(gray)
        seed_density = 1.0 - gray01 if polarity == "dark" else gray01
        gray = center_crop_from_mask(gray, normalise01(seed_density), cfg.margin)
        img = Image.fromarray(np.clip(gray, 0.0, 255.0).astype(np.uint8), mode="L")
        gray = np.asarray(img.resize((cfg.size, cfg.size), Image.LANCZOS), dtype=np.float32)

    density, silhouette, edge, shading = build_density(gray, cfg)
    save_density(cfg.out_path, density)

    preview_path = cfg.preview_path
    if preview_path is None:
        out = Path(cfg.out_path)
        preview_path = str(out.with_name(f"{out.stem}_preview{out.suffix}"))
    save_preview(preview_path, normalise01(gray), density)

    polarity = cfg.foreground
    if polarity == "auto":
        polarity = infer_foreground_polarity(gray)
    print(f"saved target density: {cfg.out_path}")
    print(f"saved preview:        {preview_path}")
    print(
        "foreground="
        f"{polarity} silhouette={cfg.silhouette_weight} edge={cfg.edge_weight} "
        f"shading={cfg.shading_weight}"
    )
    print(
        "component maxima="
        f" silhouette:{silhouette.max():.3f} edge:{edge.max():.3f} shading:{shading.max():.3f}"
    )


if __name__ == "__main__":
    main()
