"""Inverse Optimization of Strange Attractors for 2D Image Generation.

Complete implementation of the algorithm described in
"Inverse Optimization of Chaotic Attractors":

    1. 12-parameter 2D general quadratic polynomial map.
    2. Four-tier Fast Rejection Penalty System
       (divergence / fixed-point / low-period cycle / Lyapunov).
    3. Sliced Wasserstein Distance fitness against a target image.
    4. Self-contained CMA-ES with rank-mu / rank-1 covariance updates.
    5. Boundary-aware dynamic step-size dampening.
    6. Logistic-map chaotic noise injection for tunneling between
       islands of stability.

Usage:
    uv run python main.py [target_image_path] [--out best.png]

If no target is provided a synthetic pattern is generated.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# 1. Two-dimensional general quadratic polynomial map (12 parameters)
# ---------------------------------------------------------------------------
#   x' = p0 + p1 x + p2 x^2 + p3 x y + p4 y + p5 y^2
#   y' = p6 + p7 x + p8 x^2 + p9 x y + p10 y + p11 y^2
#
# All operations are vectorised across the entire CMA-ES population so a
# single Python-level step advances every individual at once.

PARAM_DIM = 12


def quadratic_map(x: np.ndarray, y: np.ndarray, p: np.ndarray):
    x2, y2, xy = x * x, y * y, x * y
    nx = (
        p[..., 0]
        + p[..., 1] * x
        + p[..., 2] * x2
        + p[..., 3] * xy
        + p[..., 4] * y
        + p[..., 5] * y2
    )
    ny = (
        p[..., 6]
        + p[..., 7] * x
        + p[..., 8] * x2
        + p[..., 9] * xy
        + p[..., 10] * y
        + p[..., 11] * y2
    )
    return nx, ny


def jacobian(x: np.ndarray, y: np.ndarray, p: np.ndarray):
    j11 = p[..., 1] + 2.0 * p[..., 2] * x + p[..., 3] * y
    j12 = p[..., 3] * x + p[..., 4] + 2.0 * p[..., 5] * y
    j21 = p[..., 7] + 2.0 * p[..., 8] * x + p[..., 9] * y
    j22 = p[..., 9] * x + p[..., 10] + 2.0 * p[..., 11] * y
    return j11, j12, j21, j22


# ---------------------------------------------------------------------------
# 2. Fast Rejection Penalty System
# ---------------------------------------------------------------------------
@dataclass
class Config:
    pop: int = 24                       # CMA-ES population (lambda)
    n_test: int = 600                   # transient testing iterations
    n_iter: int = 60_000                # rasterisation orbit length
    n_sample: int = 4096                # subsample size for SWD
    n_proj: int = 48                    # SWD random projection count
    img_size: int = 96                  # square output resolution
    max_gens: int = 80
    sigma0: float = 0.15

    escape_radius: float = 1.0e6        # Tier 1
    fp_eps: float = 1.0e-6              # Tier 2
    cycle_window: int = 64              # Tier 3 ring buffer length
    cycle_eps: float = 1.0e-5           # Tier 3 cycle equality tolerance

    rejection_threshold: float = 0.70   # boundary-aware dampening trigger
    rejection_sigma_damp: float = 0.5
    chaotic_injection_prob: float = 0.10
    chaotic_inject_frac: float = 0.25

    max_penalty: float = 1.0e6
    seed: int = 42
    out_path: str = "best_attractor.png"
    target_path: str | None = None


def transient_test(params: np.ndarray, cfg: Config):
    """Vectorised four-tier fast-rejection transient test.

    Returns
    -------
    rejected : (lam,) bool
    x, y     : (lam,) warmed state vectors
    lyap     : (lam,) maximal Lyapunov approximation
    """
    lam = params.shape[0]
    x = np.full(lam, 0.1, dtype=np.float64)
    y = np.full(lam, 0.1, dtype=np.float64)

    rejected = np.zeros(lam, dtype=bool)

    # Tangent vector for Lyapunov exponent estimation.
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    dx = np.full(lam, inv_sqrt2)
    dy = np.full(lam, inv_sqrt2)
    lyap_sum = np.zeros(lam)

    # Ring buffer for low-period cycle detection.
    W = cfg.cycle_window
    hist_x = np.zeros((W, lam))
    hist_y = np.zeros((W, lam))

    for step in range(cfg.n_test):
        nx, ny = quadratic_map(x, y, params)

        # Tier 1: numerical divergence / overflow.
        bad = (
            ~np.isfinite(nx)
            | ~np.isfinite(ny)
            | (np.abs(nx) > cfg.escape_radius)
            | (np.abs(ny) > cfg.escape_radius)
        )

        # Tier 2: fixed-point stagnation.
        fp = np.hypot(nx - x, ny - y) < cfg.fp_eps

        # Tier 3: low-period cycle via ring buffer scan.
        upto = min(step, W)
        if upto > 0:
            ddx = hist_x[:upto] - nx[None, :]
            ddy = hist_y[:upto] - ny[None, :]
            cyc = np.any(np.sqrt(ddx * ddx + ddy * ddy) < cfg.cycle_eps, axis=0)
        else:
            cyc = np.zeros(lam, dtype=bool)
        hist_x[step % W] = nx
        hist_y[step % W] = ny

        # Tier 4: Lyapunov tangent propagation (renormalised every step).
        j11, j12, j21, j22 = jacobian(x, y, params)
        ndx = j11 * dx + j12 * dy
        ndy = j21 * dx + j22 * dy
        tnorm = np.sqrt(ndx * ndx + ndy * ndy) + 1e-300
        lyap_sum += np.log(tnorm)
        dx = ndx / tnorm
        dy = ndy / tnorm

        rejected |= bad | fp | cyc

        # Freeze rejected individuals to suppress NaN propagation.
        nx = np.where(rejected, 0.0, nx)
        ny = np.where(rejected, 0.0, ny)
        x, y = nx, ny

    lyap = lyap_sum / cfg.n_test
    rejected |= ~np.isfinite(lyap)
    rejected |= lyap < 0.0  # Tier 4 final check
    return rejected, x, y, lyap


# ---------------------------------------------------------------------------
# 3. Empirical-measure rasterisation and Sliced Wasserstein Distance
# ---------------------------------------------------------------------------
def simulate_orbit(params: np.ndarray, x0: np.ndarray, y0: np.ndarray, n_iter: int):
    """Iterate the map n_iter steps for the entire surviving sub-population.

    Returns an (n_iter, lam, 2) float32 array of orbit samples.
    """
    lam = params.shape[0]
    out = np.empty((n_iter, lam, 2), dtype=np.float32)
    x = x0.astype(np.float64).copy()
    y = y0.astype(np.float64).copy()
    for step in range(n_iter):
        nx, ny = quadratic_map(x, y, params)
        bad = (
            ~np.isfinite(nx)
            | ~np.isfinite(ny)
            | (np.abs(nx) > 1.0e8)
            | (np.abs(ny) > 1.0e8)
        )
        nx = np.where(bad, 0.0, nx)
        ny = np.where(bad, 0.0, ny)
        out[step, :, 0] = nx
        out[step, :, 1] = ny
        x, y = nx, ny
    return out


def normalise_points(pts: np.ndarray, size: int):
    """Rescale a 2D point cloud into the [0, size] x [0, size] image frame."""
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
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
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
    """Monte-Carlo Sliced Wasserstein-1 between two equally sized point clouds.

    For every random projection direction the per-axis cost is the
    closed-form 1D optimal transport — a sort followed by an L1 difference.
    """
    if pts_g.shape[0] != pts_t.shape[0]:
        n = min(pts_g.shape[0], pts_t.shape[0])
        pts_g = pts_g[:n]
        pts_t = pts_t[:n]
    thetas = rng.uniform(0.0, math.pi, n_proj)
    dirs = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (K, 2)
    pg = pts_g @ dirs.T
    pt = pts_t @ dirs.T
    pg.sort(axis=0)
    pt.sort(axis=0)
    return float(np.mean(np.abs(pg - pt)))


def sample_target_points(target_density: np.ndarray, n: int, rng) -> np.ndarray:
    H, W = target_density.shape
    flat = target_density.astype(np.float64).ravel()
    s = flat.sum()
    if s <= 0:
        raise ValueError("target image density is empty")
    probs = flat / s
    idx = rng.choice(H * W, size=n, p=probs)
    py = idx // W
    px = idx % W
    px = px.astype(np.float64) + rng.random(n)
    py = py.astype(np.float64) + rng.random(n)
    return np.stack([px, py], axis=1)


# ---------------------------------------------------------------------------
# 4. Population evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate_population(params, target_pts, cfg, rng):
    lam = params.shape[0]
    rejected, x, y, _ = transient_test(params, cfg)
    fitness = np.full(lam, cfg.max_penalty, dtype=np.float64)

    survivors = np.where(~rejected)[0]
    if survivors.size == 0:
        return fitness, 1.0

    sp = params[survivors]
    sx = x[survivors]
    sy = y[survivors]
    bad_warm = ~np.isfinite(sx) | ~np.isfinite(sy)
    sx[bad_warm] = 0.1
    sy[bad_warm] = 0.1

    orbit = simulate_orbit(sp, sx, sy, cfg.n_iter)

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
        fitness[i] = sliced_wasserstein(npts, target_pts, cfg.n_proj, rng)

    return fitness, float(rejected.mean())


# ---------------------------------------------------------------------------
# 5. CMA-ES (self-contained, Hansen rank-mu/rank-1 update)
# ---------------------------------------------------------------------------
class CMAES:
    def __init__(self, x0, sigma, pop, rng):
        self.n = len(x0)
        self.m = np.asarray(x0, dtype=float).copy()
        self.sigma = float(sigma)
        self.lam = int(pop)
        self.mu = self.lam // 2

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
        D2, self.B = np.linalg.eigh(self.C)
        D2 = np.maximum(D2, 1e-30)
        self.D = np.sqrt(D2)

    def _logistic_block(self, k: int) -> np.ndarray:
        """Generate a (k, n) heavy-tailed perturbation block from the
        logistic map operating in its maximally chaotic regime r=4."""
        out = np.empty((k, self.n))
        s = self._chaos_state
        for i in range(k):
            for d in range(self.n):
                s = 4.0 * s * (1.0 - s)
                # Map uniform-ish [0,1] to a heavy-tailed perturbation in
                # roughly [-3, 3] — substitutes Gaussian sampling.
                out[i, d] = 6.0 * (s - 0.5)
        self._chaos_state = s
        return out

    def ask(self, inject_chaos: bool = False) -> np.ndarray:
        # Periodic eigen-decomposition of C.
        if (
            self.counteval - self.eigen_eval
            > self.lam / max(self.c1 + self.cmu, 1e-12) / self.n / 10.0
        ):
            self.eigen_eval = self.counteval
            self._update_eigen()

        z = self.rng.standard_normal((self.lam, self.n))
        if inject_chaos:
            k = max(1, int(round(self.lam * 0.25)))
            z[:k] = self._logistic_block(k)

        arz = z * self.D
        arx = self.m + self.sigma * (arz @ self.B.T)
        return arx

    def tell(self, arx: np.ndarray, fitness: np.ndarray) -> None:
        self.counteval += self.lam
        order = np.argsort(fitness)
        selected = arx[order[: self.mu]]

        old_m = self.m.copy()
        self.m = self.weights @ selected

        invD = np.where(self.D > 1e-12, 1.0 / self.D, 0.0)
        C_invsqrt = (self.B * invD) @ self.B.T

        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * (C_invsqrt @ (self.m - old_m) / self.sigma)
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


# ---------------------------------------------------------------------------
# 6. Main optimisation loop & utilities
# ---------------------------------------------------------------------------
def load_target_image(path: str | None, size: int) -> np.ndarray:
    """Return an HxW float32 density (high value = high desired density)."""
    if path is None:
        # Synthetic default: a hollow heart-ish curve so the demo is fun.
        img = Image.new("L", (size, size), 0)
        d = ImageDraw.Draw(img)
        d.ellipse([size * 0.18, size * 0.18, size * 0.82, size * 0.82], outline=255, width=2)
        d.line([size * 0.2, size * 0.5, size * 0.8, size * 0.5], fill=255, width=2)
        d.line([size * 0.5, size * 0.2, size * 0.5, size * 0.8], fill=255, width=2)
        return np.asarray(img, dtype=np.float32)

    img = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32)
    # Treat dark ink as density.
    a = 255.0 - a
    return np.maximum(a, 0.0)


def render_attractor(params: np.ndarray, cfg: Config, n_iter: int | None = None) -> np.ndarray:
    """Render a single parameter vector to a high-quality histogram image."""
    n_iter = n_iter or max(cfg.n_iter * 4, 200_000)
    p = params.reshape(1, PARAM_DIM)
    x = np.array([0.1])
    y = np.array([0.1])
    # Burn-in.
    for _ in range(2000):
        x, y = quadratic_map(x, y, p)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    orbit = simulate_orbit(p, x, y, n_iter)[:, 0, :].astype(np.float64)
    npts = normalise_points(orbit, cfg.img_size)
    if npts is None:
        return np.zeros((cfg.img_size, cfg.img_size), dtype=np.float64)
    return histogram_from_points(npts, cfg.img_size)


def density_to_uint8(h: np.ndarray) -> np.ndarray:
    if h.max() <= 0:
        return np.zeros_like(h, dtype=np.uint8)
    # Logarithmic stretching for nicer contrast.
    h = np.log1p(h)
    h = h / h.max()
    return (255.0 * h).astype(np.uint8)


def initial_mean() -> np.ndarray:
    """Seed CMA-ES near a known-chaotic 2D quadratic regime.

    Henon's classical attractor (a=1.4, b=0.3) lives at
        x' = 1 - 1.4 x^2 + y
        y' = 0.3 x
    which is a guaranteed chaotic point in this 12-D parameter space.
    """
    return np.array(
        [
            1.0, 0.0, -1.4, 0.0, 1.0, 0.0,   # x' coefficients
            0.0, 0.3, 0.0, 0.0, 0.0, 0.0,    # y' coefficients
        ],
        dtype=float,
    )


def optimise(cfg: Config) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    target_density = load_target_image(cfg.target_path, cfg.img_size)
    target_pts = sample_target_points(target_density, cfg.n_sample, rng)

    es = CMAES(initial_mean(), cfg.sigma0, cfg.pop, rng)

    best_loss = math.inf
    best_params = es.m.copy()
    last_rej_rate = 0.0

    for gen in range(1, cfg.max_gens + 1):
        inject = rng.random() < cfg.chaotic_injection_prob
        arx = es.ask(inject_chaos=inject)
        t0 = time.time()
        fitness, rej_rate = evaluate_population(arx, target_pts, cfg, rng)
        dt = time.time() - t0

        # Boundary-aware dynamic step-size dampening.
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


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default=None, help="Path to a target image (grayscale).")
    p.add_argument("--out", default="best_attractor.png")
    p.add_argument("--gens", type=int, default=80)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--n-iter", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return Config(
        pop=args.pop,
        max_gens=args.gens,
        img_size=args.size,
        n_iter=args.n_iter,
        seed=args.seed,
        out_path=args.out,
        target_path=args.target,
    )


def main() -> None:
    cfg = parse_args()
    print(f"Inverse Frobenius-Perron Optimisation  pop={cfg.pop} gens={cfg.max_gens} "
          f"size={cfg.img_size} n_iter={cfg.n_iter}")

    best_params, best_loss, target_density = optimise(cfg)

    print("best parameters:")
    print(np.array2string(best_params, precision=5, suppress_small=True))

    final_hist = render_attractor(best_params, cfg, n_iter=max(cfg.n_iter * 4, 250_000))
    img_rendered = density_to_uint8(final_hist)
    target_img = density_to_uint8(target_density)

    side_by_side = np.concatenate([target_img, img_rendered], axis=1)
    Image.fromarray(side_by_side).save(cfg.out_path)
    print(f"saved {cfg.out_path}  (target | attractor)  loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
