"""Inverse Optimization of Strange Attractors for 2D Image Generation.

Supports two model families:

1. ``quadratic2d``: the original 12-parameter planar quadratic map.
2. ``poly_sin3d``: a 27-parameter latent 3D map with bounded sinusoidal
   folding, projected back to 2D for scoring and rendering.

The optimisation objective can be either:

1. ``single``: the original single-scale Sliced Wasserstein distance.
2. ``multiscale``: a coarse-to-fine weighted sum of SWD terms.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Config:
    pop: int = 24
    n_test: int = 600
    n_iter: int = 60_000
    n_sample: int = 4096
    n_proj: int = 48
    img_size: int = 96
    max_gens: int = 80
    sigma0: float = 0.15

    escape_radius: float = 1.0e6
    fp_eps: float = 1.0e-6
    cycle_window: int = 64
    cycle_eps: float = 1.0e-5

    rejection_threshold: float = 0.70
    rejection_sigma_damp: float = 0.5
    chaotic_injection_prob: float = 0.10
    chaotic_inject_frac: float = 0.25

    max_penalty: float = 1.0e6
    seed: int = 42
    out_path: str = "best_attractor.png"
    target_path: str | None = None
    model_name: str = "quadratic2d"
    loss_mode: str = "single"


@dataclass
class TargetLevel:
    scale: int
    points: np.ndarray
    density: np.ndarray


@dataclass
class SupportTarget:
    scale: int
    dt_to_support: np.ndarray
    support_points: np.ndarray


class AttractorModel:
    name: str
    state_dim: int
    param_dim: int

    def step(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def project(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_mean(self) -> np.ndarray:
        raise NotImplementedError

    def initial_state(self, lam: int) -> np.ndarray:
        raise NotImplementedError


class Quadratic2DModel(AttractorModel):
    name = "quadratic2d"
    state_dim = 2
    param_dim = 12

    def step(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = state[:, 0]
        y = state[:, 1]
        x2 = x * x
        y2 = y * y
        xy = x * y
        nx = (
            params[:, 0]
            + params[:, 1] * x
            + params[:, 2] * x2
            + params[:, 3] * xy
            + params[:, 4] * y
            + params[:, 5] * y2
        )
        ny = (
            params[:, 6]
            + params[:, 7] * x
            + params[:, 8] * x2
            + params[:, 9] * xy
            + params[:, 10] * y
            + params[:, 11] * y2
        )
        return np.stack([nx, ny], axis=1)

    def jacobian(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = state[:, 0]
        y = state[:, 1]
        j11 = params[:, 1] + 2.0 * params[:, 2] * x + params[:, 3] * y
        j12 = params[:, 3] * x + params[:, 4] + 2.0 * params[:, 5] * y
        j21 = params[:, 7] + 2.0 * params[:, 8] * x + params[:, 9] * y
        j22 = params[:, 9] * x + params[:, 10] + 2.0 * params[:, 11] * y
        out = np.empty((state.shape[0], 2, 2), dtype=np.float64)
        out[:, 0, 0] = j11
        out[:, 0, 1] = j12
        out[:, 1, 0] = j21
        out[:, 1, 1] = j22
        return out

    def project(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        del params
        return state[:, :2]

    def initial_mean(self) -> np.ndarray:
        return np.array(
            [
                1.0, 0.0, -1.4, 0.0, 1.0, 0.0,
                0.0, 0.3, 0.0, 0.0, 0.0, 0.0,
            ],
            dtype=float,
        )

    def initial_state(self, lam: int) -> np.ndarray:
        return np.full((lam, 2), 0.1, dtype=np.float64)


class PolySin3DModel(AttractorModel):
    name = "poly_sin3d"
    state_dim = 3
    param_dim = 27

    def step(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        x2 = x * x
        y2 = y * y
        xy = x * y

        sx = np.sin(y + params[:, 8] * z)
        sy = np.sin(x + params[:, 17] * z)
        sz = np.sin(params[:, 23] * x + params[:, 24] * y)

        nx = (
            params[:, 0]
            + params[:, 1] * x
            + params[:, 2] * x2
            + params[:, 3] * xy
            + params[:, 4] * y
            + params[:, 5] * y2
            + params[:, 6] * z
            + params[:, 7] * sx
        )
        ny = (
            params[:, 9]
            + params[:, 10] * x
            + params[:, 11] * x2
            + params[:, 12] * xy
            + params[:, 13] * y
            + params[:, 14] * y2
            + params[:, 15] * z
            + params[:, 16] * sy
        )
        nz = (
            params[:, 18]
            + params[:, 19] * x
            + params[:, 20] * y
            + params[:, 21] * z
            + params[:, 22] * sz
        )
        return np.stack([nx, ny, nz], axis=1)

    def jacobian(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]

        cx = np.cos(y + params[:, 8] * z)
        cy = np.cos(x + params[:, 17] * z)
        cz = np.cos(params[:, 23] * x + params[:, 24] * y)

        out = np.empty((state.shape[0], 3, 3), dtype=np.float64)
        out[:, 0, 0] = params[:, 1] + 2.0 * params[:, 2] * x + params[:, 3] * y
        out[:, 0, 1] = params[:, 3] * x + params[:, 4] + params[:, 7] * cx
        out[:, 0, 2] = params[:, 6] + params[:, 7] * cx * params[:, 8]

        out[:, 1, 0] = (
            params[:, 10] + 2.0 * params[:, 11] * x + params[:, 12] * y + params[:, 16] * cy
        )
        out[:, 1, 1] = params[:, 12] * x + params[:, 13] + 2.0 * params[:, 14] * y
        out[:, 1, 2] = params[:, 15] + params[:, 16] * cy * params[:, 17]

        out[:, 2, 0] = params[:, 19] + params[:, 22] * cz * params[:, 23]
        out[:, 2, 1] = params[:, 20] + params[:, 22] * cz * params[:, 24]
        out[:, 2, 2] = params[:, 21]
        return out

    def project(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        u = x + params[:, 25] * z
        v = y + params[:, 26] * z
        return np.stack([u, v], axis=1)

    def initial_mean(self) -> np.ndarray:
        return np.array(
            [
                1.0, 0.0, -1.4, 0.0, 1.0, 0.0, 0.0, 0.15, 0.40,
                0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12, -0.35,
                0.0, 0.15, -0.10, 0.25, 0.10, 0.80, -0.60, 0.30, 0.20,
            ],
            dtype=float,
        )

    def initial_state(self, lam: int) -> np.ndarray:
        out = np.zeros((lam, 3), dtype=np.float64)
        out[:, 0] = 0.1
        out[:, 1] = 0.1
        return out


MODELS = {
    "quadratic2d": Quadratic2DModel(),
    "poly_sin3d": PolySin3DModel(),
}


def get_model(name: str) -> AttractorModel:
    try:
        return MODELS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}") from exc


def load_target_image(path: str | None, size: int) -> np.ndarray:
    """Return an HxW float32 density (high value = high desired density)."""
    if path is None:
        img = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(img)
        d.ellipse([size * 0.18, size * 0.18, size * 0.82, size * 0.82], outline=255, width=2)
        d.line([size * 0.2, size * 0.5, size * 0.8, size * 0.5], fill=255, width=2)
        d.line([size * 0.5, size * 0.2, size * 0.5, size * 0.8], fill=255, width=2)
        return np.asarray(img, dtype=np.float32)

    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    a = 255.0 - a
    return np.maximum(a, 0.0)


def sample_target_points(target_density: np.ndarray, n: int, rng) -> np.ndarray:
    h, w = target_density.shape
    flat = target_density.astype(np.float64).ravel()
    s = flat.sum()
    if s <= 0:
        raise ValueError("target image density is empty")
    probs = flat / s
    idx = rng.choice(h * w, size=n, p=probs)
    py = idx // w
    px = idx % w
    px = px.astype(np.float64) + rng.random(n)
    py = py.astype(np.float64) + rng.random(n)
    return np.stack([px, py], axis=1)


def multiscale_spec(img_size: int) -> list[int]:
    raw = [32, 64, 128]
    scales = [scale for scale in raw if scale <= img_size]
    if not scales:
        scales = [img_size]
    return scales


def edt_1d(f: np.ndarray) -> np.ndarray:
    n = f.shape[0]
    v = np.zeros(n, dtype=np.int32)
    z = np.zeros(n + 1, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        out[q] = dq * dq + f[v[k]]
    return out


def distance_transform(mask: np.ndarray) -> np.ndarray:
    inf = 1.0e12
    f = np.where(mask, 0.0, inf).astype(np.float64)
    tmp = np.empty_like(f)
    for x in range(f.shape[1]):
        tmp[:, x] = edt_1d(f[:, x])
    out = np.empty_like(tmp)
    for y in range(tmp.shape[0]):
        out[y, :] = edt_1d(tmp[y, :])
    return np.sqrt(out)


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y_src0 = max(0, -dy)
            y_src1 = min(mask.shape[0], mask.shape[0] - dy)
            x_src0 = max(0, -dx)
            x_src1 = min(mask.shape[1], mask.shape[1] - dx)
            y_dst0 = max(0, dy)
            y_dst1 = min(mask.shape[0], mask.shape[0] + dy)
            x_dst0 = max(0, dx)
            x_dst1 = min(mask.shape[1], mask.shape[1] + dx)
            out[y_dst0:y_dst1, x_dst0:x_dst1] |= mask[y_src0:y_src1, x_src0:x_src1]
    return out


def density_support_mask(density: np.ndarray) -> np.ndarray:
    if density.max() <= 0:
        return np.zeros_like(density, dtype=bool)
    norm = density / density.max()
    mask = norm > 0.15
    return dilate_mask(mask, radius=1)


def build_target_levels(cfg: Config, rng) -> tuple[np.ndarray, list[TargetLevel], SupportTarget | None]:
    full_density = load_target_image(cfg.target_path, cfg.img_size)
    if cfg.loss_mode == "single":
        return (
            full_density,
            [TargetLevel(cfg.img_size, sample_target_points(full_density, cfg.n_sample, rng), full_density)],
            None,
        )

    levels: list[TargetLevel] = []
    for scale in multiscale_spec(cfg.img_size):
        density = load_target_image(cfg.target_path, scale)
        pts = sample_target_points(density, cfg.n_sample, rng)
        levels.append(TargetLevel(scale=scale, points=pts, density=density))

    support_scale = min(64, max(level.scale for level in levels))
    support_density = next(level.density for level in levels if level.scale == support_scale)
    support_mask = density_support_mask(support_density)
    support_points = np.argwhere(support_mask).astype(np.int32)
    support_target = SupportTarget(
        scale=support_scale,
        dt_to_support=distance_transform(support_mask),
        support_points=support_points,
    )
    return full_density, levels, support_target


def normalise_points(pts: np.ndarray, size: int):
    finite = np.isfinite(pts).all(axis=1)
    if finite.sum() < 16:
        return None
    pts = pts[finite]
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    rx = xmax - xmin
    ry = ymax - ymin
    if rx < 1e-9 or ry < 1e-9:
        return None
    s = max(rx, ry)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - cx) / s * (0.9 * size) + 0.5 * size
    out[:, 1] = (pts[:, 1] - cy) / s * (0.9 * size) + 0.5 * size
    return out


def histogram_from_points(pts: np.ndarray, size: int):
    px = np.clip(pts[:, 0].astype(np.int32), 0, size - 1)
    py = np.clip(pts[:, 1].astype(np.int32), 0, size - 1)
    h = np.zeros((size, size), dtype=np.float64)
    np.add.at(h, (py, px), 1.0)
    return h


def sliced_wasserstein(pts_g: np.ndarray, pts_t: np.ndarray, n_proj: int, rng):
    if pts_g.shape[0] != pts_t.shape[0]:
        n = min(pts_g.shape[0], pts_t.shape[0])
        pts_g = pts_g[:n]
        pts_t = pts_t[:n]
    thetas = rng.uniform(0.0, math.pi, n_proj)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    pg = pts_g @ dirs.T
    pt = pts_t @ dirs.T
    pg.sort(axis=0)
    pt.sort(axis=0)
    return float(np.mean(np.abs(pg - pt)))


def state_bad_mask(state: np.ndarray, escape_radius: float) -> np.ndarray:
    return (~np.isfinite(state)).any(axis=1) | (np.max(np.abs(state), axis=1) > escape_radius)


def transient_test(model: AttractorModel, params: np.ndarray, cfg: Config):
    """Vectorised fast-rejection transient test on hidden state."""
    lam = params.shape[0]
    state = model.initial_state(lam)
    rejected = np.zeros(lam, dtype=bool)

    tangent = np.full((lam, model.state_dim), 1.0 / math.sqrt(model.state_dim), dtype=np.float64)
    lyap_sum = np.zeros(lam, dtype=np.float64)

    hist = np.zeros((cfg.cycle_window, lam, model.state_dim), dtype=np.float64)

    for step in range(cfg.n_test):
        next_state = model.step(state, params)

        bad = state_bad_mask(next_state, cfg.escape_radius)
        fp = np.linalg.norm(next_state - state, axis=1) < cfg.fp_eps

        upto = min(step, cfg.cycle_window)
        if upto > 0:
            diff = hist[:upto] - next_state[None, :, :]
            cyc = np.any(np.sqrt(np.sum(diff * diff, axis=2)) < cfg.cycle_eps, axis=0)
        else:
            cyc = np.zeros(lam, dtype=bool)
        hist[step % cfg.cycle_window] = next_state

        jac = model.jacobian(state, params)
        next_tangent = np.einsum("lij,lj->li", jac, tangent)
        tnorm = np.linalg.norm(next_tangent, axis=1) + 1e-300
        lyap_sum += np.log(tnorm)
        tangent = next_tangent / tnorm[:, None]

        rejected |= bad | fp | cyc
        state = np.where(rejected[:, None], 0.0, next_state)

    lyap = lyap_sum / cfg.n_test
    rejected |= ~np.isfinite(lyap)
    rejected |= lyap < 0.0
    return rejected, state, lyap


def simulate_orbit(model: AttractorModel, params: np.ndarray, state0: np.ndarray, n_iter: int):
    """Iterate hidden state, storing projected 2D points only."""
    lam = params.shape[0]
    out = np.empty((n_iter, lam, 2), dtype=np.float32)
    state = state0.astype(np.float64).copy()

    for step in range(n_iter):
        next_state = model.step(state, params)
        bad = state_bad_mask(next_state, 1.0e8)
        next_state = np.where(bad[:, None], 0.0, next_state)
        proj = model.project(next_state, params)
        proj_bad = (~np.isfinite(proj)).any(axis=1)
        if np.any(proj_bad):
            proj = np.where(proj_bad[:, None], 0.0, proj)
            next_state = np.where(proj_bad[:, None], 0.0, next_state)
        out[step] = proj.astype(np.float32)
        state = next_state
    return out


def multiscale_weights(num_levels: int, progress: float) -> np.ndarray:
    if num_levels <= 1:
        return np.array([1.0], dtype=np.float64)
    if num_levels == 2:
        if progress < 1.0 / 3.0:
            return np.array([0.70, 0.30], dtype=np.float64)
        if progress < 2.0 / 3.0:
            return np.array([0.40, 0.60], dtype=np.float64)
        return np.array([0.20, 0.80], dtype=np.float64)
    if progress < 1.0 / 3.0:
        return np.array([0.60, 0.30, 0.10], dtype=np.float64)
    if progress < 2.0 / 3.0:
        return np.array([0.25, 0.45, 0.30], dtype=np.float64)
    return np.array([0.10, 0.30, 0.60], dtype=np.float64)


def support_loss(pts: np.ndarray, support_target: SupportTarget) -> float:
    scale = support_target.scale
    px = np.clip(pts[:, 0].astype(np.int32), 0, scale - 1)
    py = np.clip(pts[:, 1].astype(np.int32), 0, scale - 1)
    forward = float(np.mean(support_target.dt_to_support[py, px]))

    gen_mask = np.zeros((scale, scale), dtype=bool)
    gen_mask[py, px] = True
    gen_mask = dilate_mask(gen_mask, radius=1)
    gen_dt = distance_transform(gen_mask)
    target_py = support_target.support_points[:, 0]
    target_px = support_target.support_points[:, 1]
    reverse = float(np.mean(gen_dt[target_py, target_px]))
    return forward + 0.75 * reverse


def score_points_against_targets(
    pts: np.ndarray,
    target_levels: list[TargetLevel],
    support_target: SupportTarget | None,
    cfg: Config,
    rng,
    progress: float,
) -> float:
    loss = 0.0
    weights = multiscale_weights(len(target_levels), progress)
    for weight, level in zip(weights, target_levels, strict=True):
        if level.scale == cfg.img_size:
            scaled_pts = pts
        else:
            scaled_pts = pts * (level.scale / cfg.img_size)
        loss += float(weight) * sliced_wasserstein(scaled_pts, level.points, cfg.n_proj, rng)
    if support_target is not None:
        support_pts = pts * (support_target.scale / cfg.img_size)
        loss += 0.35 * support_loss(support_pts, support_target)
    return float(loss)


def evaluate_population(model: AttractorModel, params, target_levels, support_target, cfg, rng, progress: float):
    lam = params.shape[0]
    rejected, state, _ = transient_test(model, params, cfg)
    fitness = np.full(lam, cfg.max_penalty, dtype=np.float64)

    survivors = np.where(~rejected)[0]
    if survivors.size == 0:
        return fitness, 1.0

    sp = params[survivors]
    ss = state[survivors].copy()
    bad_warm = (~np.isfinite(ss)).any(axis=1)
    if np.any(bad_warm):
        ss[bad_warm] = model.initial_state(int(np.sum(bad_warm)))

    orbit = simulate_orbit(model, sp, ss, cfg.n_iter)

    for j, i in enumerate(survivors):
        pts = orbit[:, j, :].astype(np.float64)
        npts = normalise_points(pts, cfg.img_size)
        if npts is None:
            continue
        if npts.shape[0] > cfg.n_sample:
            sel = rng.choice(npts.shape[0], size=cfg.n_sample, replace=False)
            npts = npts[sel]
        elif npts.shape[0] < cfg.n_sample:
            sel = rng.choice(npts.shape[0], size=cfg.n_sample, replace=True)
            npts = npts[sel]
        fitness[i] = score_points_against_targets(
            npts,
            target_levels,
            support_target,
            cfg,
            rng,
            progress,
        )

    return fitness, float(rejected.mean())


class CMAES:
    def __init__(self, x0, sigma, pop, rng, chaotic_inject_frac: float = 0.25):
        self.n = len(x0)
        self.m = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma)
        self.lam = int(pop)
        self.mu = self.lam // 2
        self.chaotic_inject_frac = float(chaotic_inject_frac)

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        w /= w.sum()
        self.weights = w
        self.mueff = 1.0 / float(np.sum(w * w))

        n = self.n
        me = self.mueff
        self.cc = (4.0 + me / n) / (n + 4.0 + 2.0 * me / n)
        self.cs = (me + 2.0) / (n + me + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + me)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (me - 2.0 + 1.0 / me) / ((n + 2.0) ** 2 + me),
        )
        self.damps = (
            1.0 + 2.0 * max(0.0, math.sqrt((me - 1.0) / (n + 1.0)) - 1.0) + self.cs
        )

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.eigen_eval = 0
        self.counteval = 0
        self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.rng = rng
        self._chaos_state = float(rng.uniform(0.05, 0.95))

    def _update_eigen(self):
        self.C = 0.5 * (self.C + self.C.T)
        d2, self.B = np.linalg.eigh(self.C)
        d2 = np.maximum(d2, 1e-30)
        self.D = np.sqrt(d2)

    def _logistic_block(self, k: int) -> np.ndarray:
        out = np.empty((k, self.n))
        s = self._chaos_state
        for i in range(k):
            for d in range(self.n):
                s = 4.0 * s * (1.0 - s)
                out[i, d] = 6.0 * (s - 0.5)
        self._chaos_state = s
        return out

    def ask(self, inject_chaos: bool = False) -> np.ndarray:
        if (
            self.counteval - self.eigen_eval
            > self.lam / max(self.c1 + self.cmu, 1e-12) / self.n / 10.0
        ):
            self.eigen_eval = self.counteval
            self._update_eigen()

        z = self.rng.standard_normal((self.lam, self.n))
        if inject_chaos:
            k = max(1, int(round(self.lam * self.chaotic_inject_frac)))
            z[:k] = self._logistic_block(k)

        arz = z * self.D
        return self.m + self.sigma * (arz @ self.B.T)

    def tell(self, arx: np.ndarray, fitness: np.ndarray) -> None:
        self.counteval += self.lam
        order = np.argsort(fitness)
        selected = arx[order[: self.mu]]

        old_m = self.m.copy()
        self.m = self.weights @ selected

        inv_d = np.where(self.D > 1e-12, 1.0 / self.D, 0.0)
        c_invsqrt = (self.B * inv_d) @ self.B.T

        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * (c_invsqrt @ (self.m - old_m) / self.sigma)
        norm_ps = float(np.linalg.norm(self.ps))

        gen = self.counteval / self.lam
        denom = math.sqrt(max(1.0 - (1.0 - self.cs) ** (2.0 * gen), 1e-30))
        hs = float(norm_ps / denom / self.chiN < (1.4 + 2.0 / (self.n + 1.0)))

        self.pc = (1.0 - self.cc) * self.pc + hs * math.sqrt(
            self.cc * (2.0 - self.cc) * self.mueff
        ) * (self.m - old_m) / self.sigma

        artmp = (selected - old_m) / self.sigma
        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1
            * (np.outer(self.pc, self.pc) + (1.0 - hs) * self.cc * (2.0 - self.cc) * self.C)
            + self.cmu * (artmp.T * self.weights) @ artmp
        )

        self.sigma *= math.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1.0))
        self.sigma = float(np.clip(self.sigma, 1e-8, 1e3))


def density_to_uint8(h: np.ndarray) -> np.ndarray:
    if h.max() <= 0:
        return np.zeros_like(h, dtype=np.uint8)
    h = np.log1p(h)
    h = h / h.max()
    return (255.0 * h).astype(np.uint8)


def render_attractor(model: AttractorModel, params: np.ndarray, cfg: Config, n_iter: int | None = None) -> np.ndarray:
    n_iter = n_iter or max(cfg.n_iter * 4, 200_000)
    p = params.reshape(1, model.param_dim)
    state = model.initial_state(1)
    for _ in range(2000):
        state = model.step(state, p)
        bad = state_bad_mask(state, 1.0e8)
        state = np.where(bad[:, None], 0.0, state)
    orbit = simulate_orbit(model, p, state, n_iter)[:, 0, :].astype(np.float64)
    npts = normalise_points(orbit, cfg.img_size)
    if npts is None:
        return np.zeros((cfg.img_size, cfg.img_size), dtype=np.float64)
    return histogram_from_points(npts, cfg.img_size)


def optimise(cfg: Config) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    model = get_model(cfg.model_name)
    target_density, target_levels, support_target = build_target_levels(cfg, rng)

    es = CMAES(
        model.initial_mean(),
        cfg.sigma0,
        cfg.pop,
        rng,
        chaotic_inject_frac=cfg.chaotic_inject_frac,
    )

    best_loss = math.inf
    best_params = es.m.copy()
    last_rej_rate = 0.0

    for gen in range(1, cfg.max_gens + 1):
        inject = rng.random() < cfg.chaotic_injection_prob
        arx = es.ask(inject_chaos=inject)
        t0 = time.time()
        fitness, rej_rate = evaluate_population(
            model,
            arx,
            target_levels,
            support_target,
            cfg,
            rng,
            progress=(gen - 1) / max(cfg.max_gens - 1, 1),
        )
        dt = time.time() - t0

        if rej_rate > cfg.rejection_threshold:
            es.sigma *= cfg.rejection_sigma_damp

        es.tell(arx, fitness)

        gen_best = float(np.min(fitness))
        if gen_best < best_loss:
            best_loss = gen_best
            best_params = arx[int(np.argmin(fitness))].copy()

        last_rej_rate = rej_rate
        print(
            f"gen {gen:3d}  best={best_loss:9.4f}  gen_best={gen_best:9.4f}  "
            f"sigma={es.sigma:8.4f}  rej={rej_rate*100:5.1f}%  "
            f"chaos={'Y' if inject else '.'}  ({dt:5.1f}s)",
            flush=True,
        )

    print(f"final best loss={best_loss:.4f}  rejection={last_rej_rate*100:.1f}%")
    return best_params, best_loss, target_density


def resolve_model_defaults(
    model_name: str,
    pop: int | None,
    sigma0: float | None,
    n_test: int | None,
    chaotic_injection_prob: float | None,
) -> tuple[int, float, int, float]:
    if model_name == "poly_sin3d":
        return (
            32 if pop is None else pop,
            0.08 if sigma0 is None else sigma0,
            800 if n_test is None else n_test,
            0.15 if chaotic_injection_prob is None else chaotic_injection_prob,
        )
    return (
        24 if pop is None else pop,
        0.15 if sigma0 is None else sigma0,
        600 if n_test is None else n_test,
        0.10 if chaotic_injection_prob is None else chaotic_injection_prob,
    )


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default=None, help="Path to a target image (grayscale).")
    p.add_argument("--out", default="best_attractor.png")
    p.add_argument("--gens", type=int, default=80)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--n-iter", type=int, default=60_000)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--sigma0", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=sorted(MODELS), default="quadratic2d")
    p.add_argument("--loss-mode", choices=["single", "multiscale"], default="single")
    p.add_argument("--chaotic-injection-prob", type=float, default=None)
    args = p.parse_args()

    pop, sigma0, n_test, chaotic_prob = resolve_model_defaults(
        args.model,
        args.pop,
        args.sigma0,
        args.n_test,
        args.chaotic_injection_prob,
    )
    return Config(
        pop=pop,
        n_test=n_test,
        max_gens=args.gens,
        img_size=args.size,
        n_iter=args.n_iter,
        sigma0=sigma0,
        chaotic_injection_prob=chaotic_prob,
        seed=args.seed,
        out_path=args.out,
        target_path=args.target,
        model_name=args.model,
        loss_mode=args.loss_mode,
    )


def main() -> None:
    cfg = parse_args()
    print(
        "Inverse Frobenius-Perron Optimisation  "
        f"model={cfg.model_name} loss={cfg.loss_mode} pop={cfg.pop} gens={cfg.max_gens} "
        f"size={cfg.img_size} n_iter={cfg.n_iter}"
    )

    model = get_model(cfg.model_name)
    best_params, best_loss, target_density = optimise(cfg)

    print("best parameters:")
    print(np.array2string(best_params, precision=5, suppress_small=True))

    final_hist = render_attractor(model, best_params, cfg, n_iter=max(cfg.n_iter * 4, 250_000))
    img_rendered = density_to_uint8(final_hist)
    target_img = density_to_uint8(target_density)

    side_by_side = np.concatenate([target_img, img_rendered], axis=1)
    Image.fromarray(side_by_side).save(cfg.out_path)
    print(f"saved {cfg.out_path}  (target | attractor)  loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
