from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np


type Array = np.ndarray


class CurveFn(Protocol):
    def __call__(self, s: float, /) -> Sequence[float]: ...


@dataclass(slots=True)
class GuidedSegment:
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    max_projection_error: float
    mean_projection_error: float


@dataclass(slots=True)
class OrbitRecord:
    time: Array
    abstract: Array
    spatial: Array


@dataclass(slots=True)
class AbstractDynamics:
    omega: float = 1.0
    track_fraction: float = 0.6
    track_rate: float = 2.2980031125259206
    pretrack_rate: float = 1.224370079764777
    global_u_damping: float = 0.11639980478639775
    global_v_damping: float = 0.1190180350645868
    kick_gain: float = 63.06326043147467
    drift_gain: float = 7.065469332518134
    stretch_gain: float = 2.8981635151294833
    nonlinearity: float = 36.516743370527756
    transition_width: float = 0.04
    kick_location: float = 0.22
    drift_location: float = 0.48
    stretch_location: float = 0.76
    kick_width: float = 0.05
    drift_width: float = 0.05
    stretch_width: float = 0.06

    def __call__(self, state: Array) -> Array:
        theta, u, v = np.asarray(state, dtype=float)
        theta_mod = theta % 1.0
        track_weight = periodic_interval_gate(
            theta_mod,
            start=0.0,
            end=self.track_fraction,
            transition=self.transition_width,
        )
        pretrack_weight = periodic_bump(theta_mod, 0.97, 0.03)

        return_fraction = max(1.0 - self.track_fraction, 1e-6)
        kick_center = (self.track_fraction + self.kick_location * return_fraction) % 1.0
        drift_center = (self.track_fraction + self.drift_location * return_fraction) % 1.0
        stretch_center = (self.track_fraction + self.stretch_location * return_fraction) % 1.0
        kick_weight = periodic_bump(theta_mod, kick_center, self.kick_width)
        drift_weight = periodic_bump(theta_mod, drift_center, self.drift_width)
        stretch_weight = periodic_bump(theta_mod, stretch_center, self.stretch_width)

        du = (
            -(self.global_u_damping + self.track_rate * track_weight + self.pretrack_rate * pretrack_weight) * u
            + self.drift_gain * drift_weight * v
            + self.stretch_gain * stretch_weight * u
        )
        dv = (
            -(self.global_v_damping + self.track_rate * track_weight + self.pretrack_rate * pretrack_weight) * v
            + self.kick_gain * kick_weight * np.sin(self.nonlinearity * u)
        )
        return np.array([self.omega, du, dv], dtype=float)


@dataclass(slots=True)
class TubeEmbedding:
    centerline: Array
    normal1: Array
    normal2: Array
    track_fraction: float

    def __post_init__(self) -> None:
        if self.centerline.ndim != 2 or self.centerline.shape[1] != 3:
            raise ValueError("centerline must have shape (n, 3)")
        if self.normal1.shape != self.centerline.shape or self.normal2.shape != self.centerline.shape:
            raise ValueError("frame arrays must match centerline shape")
        if not (0.0 < self.track_fraction < 1.0):
            raise ValueError("track_fraction must lie in (0, 1)")
        self.num_samples = len(self.centerline)
        self.step = 1.0 / self.num_samples

    num_samples: int = 0
    step: float = 0.0

    def _interpolate(self, samples: Array, theta: float) -> Array:
        position = (theta % 1.0) * self.num_samples
        lower = int(np.floor(position)) % self.num_samples
        upper = (lower + 1) % self.num_samples
        weight = position - np.floor(position)
        return (1.0 - weight) * samples[lower] + weight * samples[upper]

    def center(self, theta: float) -> Array:
        return self._interpolate(self.centerline, theta)

    def frame(self, theta: float) -> tuple[Array, Array]:
        n1 = self._interpolate(self.normal1, theta)
        n1 /= max(np.linalg.norm(n1), 1e-12)
        n2 = self._interpolate(self.normal2, theta)
        n2 -= np.dot(n2, n1) * n1
        n2 /= max(np.linalg.norm(n2), 1e-12)
        return n1, n2

    def encode(self, theta: float, u: float, v: float) -> Array:
        center = self.center(theta)
        normal1, normal2 = self.frame(theta)
        return center + u * normal1 + v * normal2

    def decode(self, point: Sequence[float]) -> Array:
        point_array = np.asarray(point, dtype=float)
        anchors = self.centerline
        nxt = np.roll(self.centerline, -1, axis=0)
        segments = nxt - anchors
        squared_lengths = np.sum(segments * segments, axis=1)
        squared_lengths = np.where(squared_lengths > 1e-12, squared_lengths, 1e-12)

        deltas = point_array - anchors
        projections = np.sum(deltas * segments, axis=1) / squared_lengths
        projections = np.clip(projections, 0.0, 1.0)
        feet = anchors + projections[:, None] * segments
        distances = np.linalg.norm(point_array - feet, axis=1)
        index = int(np.argmin(distances))
        theta = (index + float(projections[index])) / self.num_samples

        normal1, normal2 = self.frame(theta)
        offset = point_array - feet[index]
        u = float(np.dot(offset, normal1))
        v = float(np.dot(offset, normal2))
        return np.array([theta, u, v], dtype=float)

    def jacobian(self, theta: float, u: float, v: float, epsilon: float = 1e-5) -> Array:
        d_theta = (self.encode(theta + epsilon, u, v) - self.encode(theta - epsilon, u, v)) / (2.0 * epsilon)
        normal1, normal2 = self.frame(theta)
        return np.column_stack([d_theta, normal1, normal2])


@dataclass(slots=True)
class GuidedChaoticField:
    vector_field: Callable[[Array], Array]
    abstract_field: Callable[[Array], Array]
    embedding: TubeEmbedding
    orbit: OrbitRecord
    guided_segment: GuidedSegment
    curve_samples: Array
    metadata: dict[str, float | int | str]

    def __call__(self, state: Sequence[float]) -> Array:
        return self.vector_field(np.asarray(state, dtype=float))

    def project(self, states: Array) -> Array:
        states_array = np.asarray(states, dtype=float)
        return states_array[..., :2]

    def abstract_to_spatial(self, state: Sequence[float]) -> Array:
        theta, u, v = np.asarray(state, dtype=float)
        return self.embedding.encode(theta, u, v)

    def spatial_to_abstract(self, state: Sequence[float]) -> Array:
        return self.embedding.decode(state)

    def simulate(
        self,
        total_time: float,
        dt: float,
        initial_abstract_state: Sequence[float] | None = None,
    ) -> OrbitRecord:
        if dt <= 0:
            raise ValueError("dt must be positive")
        steps = max(1, int(np.ceil(total_time / dt)))
        initial_state = (
            self.orbit.abstract[0]
            if initial_abstract_state is None
            else np.asarray(initial_abstract_state, dtype=float)
        )
        time, abstract = rk4_integrate(self.abstract_field, initial_state, dt=dt, steps=steps)
        spatial = np.array([self.abstract_to_spatial(state) for state in abstract], dtype=float)
        return OrbitRecord(time=time, abstract=abstract, spatial=spatial)

    def guided_projection(self) -> Array:
        segment = self.orbit.spatial[
            self.guided_segment.start_index : self.guided_segment.end_index + 1
        ]
        return self.project(segment)


def rk4_integrate(
    field: Callable[[Array], Array],
    initial_state: Sequence[float],
    *,
    dt: float,
    steps: int,
) -> tuple[Array, Array]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    state = np.asarray(initial_state, dtype=float)
    trajectory = np.empty((steps + 1, 3), dtype=float)
    time = np.linspace(0.0, dt * steps, steps + 1, dtype=float)
    trajectory[0] = state

    for index in range(steps):
        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[index + 1] = state

    return time, trajectory


def periodic_bump(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError("width must be positive")
    distance = ((theta - center + 0.5) % 1.0) - 0.5
    return float(np.exp(-((distance / width) ** 4)))


def smoothstep(x: float) -> float:
    clipped = float(np.clip(x, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def periodic_interval_gate(theta: float, *, start: float, end: float, transition: float) -> float:
    if transition <= 0:
        raise ValueError("transition must be positive")
    x = theta % 1.0
    y = (x - start) % 1.0
    span = (end - start) % 1.0
    if span <= 0:
        span = 1.0
    if y <= span:
        return 1.0
    if y < span + transition:
        return 1.0 - smoothstep((y - span) / transition)
    if y > 1.0 - transition:
        return smoothstep((y - (1.0 - transition)) / transition)
    return 0.0


def _resample_polyline(points: Array, num_samples: int) -> Array:
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if len(points) < 2:
        raise ValueError("at least two points are required")

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]
    if total <= 1e-12:
        return np.repeat(points[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total, num_samples, dtype=float)
    result = np.empty((num_samples, points.shape[1]), dtype=float)
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


def _sample_curve(gamma: CurveFn | Sequence[Sequence[float]] | Array, num_samples: int) -> Array:
    if callable(gamma):
        samples = np.array(
            [gamma(float(s)) for s in np.linspace(0.0, 1.0, num_samples)],
            dtype=float,
        )
    else:
        samples = np.asarray(gamma, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("array-valued gamma must have shape (n, 2)")
        if len(samples) != num_samples:
            samples = _resample_polyline(samples, num_samples)
    if samples.shape != (num_samples, 2):
        raise ValueError("sampled curve must have shape (num_samples, 2)")
    return samples


def _normalize(vector: Array) -> Array:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("encountered near-zero vector during normalization")
    return vector / norm


def _parallel_transport_frame(centerline: Array) -> tuple[Array, Array]:
    tangent = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangent = np.array([_normalize(vector) for vector in tangent], dtype=float)

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(reference, tangent[0])) > 0.85:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    normal1 = np.empty_like(centerline)
    normal2 = np.empty_like(centerline)
    first = reference - np.dot(reference, tangent[0]) * tangent[0]
    normal1[0] = _normalize(first)
    normal2[0] = _normalize(np.cross(tangent[0], normal1[0]))

    for index in range(1, len(centerline)):
        projected = normal1[index - 1] - np.dot(normal1[index - 1], tangent[index]) * tangent[index]
        if np.linalg.norm(projected) <= 1e-10:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(np.dot(fallback, tangent[index])) > 0.85:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            projected = fallback - np.dot(fallback, tangent[index]) * tangent[index]
        normal1[index] = _normalize(projected)
        normal2[index] = _normalize(np.cross(tangent[index], normal1[index]))

    return normal1, normal2


def _cubic_bezier(points: Sequence[Array], num_samples: int) -> Array:
    p0, p1, p2, p3 = [np.asarray(point, dtype=float) for point in points]
    t = np.linspace(0.0, 1.0, num_samples, dtype=float)
    omt = 1.0 - t
    return (
        (omt**3)[:, None] * p0
        + (3.0 * omt * omt * t)[:, None] * p1
        + (3.0 * omt * t * t)[:, None] * p2
        + (t**3)[:, None] * p3
    )


def _build_closed_centerline(
    curve_samples: Array,
    *,
    track_height: float,
    return_height: float,
    return_samples: int,
) -> tuple[Array, float]:
    z_track = np.linspace(0.0, track_height, len(curve_samples), dtype=float)
    track = np.column_stack([curve_samples, z_track])

    start = track[0]
    end = track[-1]
    start_tangent = _normalize(track[1] - track[0])
    end_tangent = _normalize(track[-1] - track[-2])

    planar_scale = max(float(np.ptp(curve_samples[:, 0])), float(np.ptp(curve_samples[:, 1])), 1.0)
    tangent_scale = 0.35 * planar_scale
    lift = np.array([0.0, 0.0, return_height], dtype=float)

    control1 = end + tangent_scale * end_tangent + lift
    control2 = start - tangent_scale * start_tangent + lift
    return_path = _cubic_bezier([end, control1, control2, start], return_samples)

    centerline = np.vstack([track, return_path[1:-1]])
    track_fraction = (len(track) - 1) / len(centerline)
    return centerline, track_fraction


def _interpolate_curve_samples(samples: Array, parameters: Array) -> Array:
    clipped = np.clip(np.asarray(parameters, dtype=float), 0.0, 1.0)
    scaled = clipped * (len(samples) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.clip(lower + 1, 0, len(samples) - 1)
    weight = scaled - lower
    return (1.0 - weight)[:, None] * samples[lower] + weight[:, None] * samples[upper]


def _segment_error(
    segment_projection: Array,
    target_curve: Array,
    local_parameters: Array,
) -> tuple[float, float]:
    target = _interpolate_curve_samples(target_curve, local_parameters)
    error = np.linalg.norm(segment_projection - target, axis=1)
    return float(np.max(error)), float(np.mean(error))


def _find_guided_segment(
    orbit: OrbitRecord,
    target_curve: Array,
    track_fraction: float,
) -> GuidedSegment:
    theta = orbit.abstract[:, 0] % 1.0
    cycle = np.floor(orbit.abstract[:, 0]).astype(int)
    unique_cycles = np.unique(cycle)

    best: GuidedSegment | None = None
    for cycle_id in unique_cycles:
        mask = (cycle == cycle_id) & (theta <= track_fraction)
        indices = np.flatnonzero(mask)
        if len(indices) < 8:
            continue
        projection = orbit.spatial[indices, :2]
        local_parameters = np.clip(theta[indices] / track_fraction, 0.0, 1.0)
        max_error, mean_error = _segment_error(
            projection,
            target_curve,
            local_parameters,
        )
        candidate = GuidedSegment(
            start_index=int(indices[0]),
            end_index=int(indices[-1]),
            start_time=float(orbit.time[indices[0]]),
            end_time=float(orbit.time[indices[-1]]),
            max_projection_error=max_error,
            mean_projection_error=mean_error,
        )
        if best is None or candidate.max_projection_error < best.max_projection_error:
            best = candidate

    if best is None:
        raise RuntimeError("failed to identify a guided track segment")
    return best


def _estimate_separation_growth(
    field: Callable[[Array], Array],
    *,
    initial_state: Sequence[float],
    perturbation: Sequence[float],
    dt: float,
    steps: int,
) -> tuple[float, float]:
    _, reference = rk4_integrate(field, initial_state, dt=dt, steps=steps)
    _, perturbed = rk4_integrate(
        field,
        np.asarray(initial_state, dtype=float) + np.asarray(perturbation, dtype=float),
        dt=dt,
        steps=steps,
    )
    distance = np.linalg.norm(reference[:, 1:] - perturbed[:, 1:], axis=1)
    baseline = max(float(distance[0]), 1e-12)
    return float(np.max(distance) / baseline), float(distance[-1] / baseline)


def build_guided_chaotic_field(
    gamma: CurveFn | Sequence[Sequence[float]] | Array,
    *,
    num_curve_samples: int = 140,
    total_time: float = 35.0,
    transient_time: float = 8.0,
    dt: float = 0.005,
    track_height: float = 2.0,
    return_height: float = 4.0,
    return_samples: int | None = None,
    initial_abstract_state: Sequence[float] = (0.0, 0.03, -0.02),
    dynamics: AbstractDynamics | None = None,
) -> GuidedChaoticField:
    if num_curve_samples < 48:
        raise ValueError("num_curve_samples must be at least 48")
    if total_time <= transient_time:
        raise ValueError("total_time must be larger than transient_time")
    if dt <= 0:
        raise ValueError("dt must be positive")

    curve_samples = _sample_curve(gamma, num_curve_samples)
    return_count = (
        return_samples
        if return_samples is not None
        else max(180, int(np.ceil(1.15 * num_curve_samples)))
    )
    centerline, track_fraction = _build_closed_centerline(
        curve_samples,
        track_height=track_height,
        return_height=return_height,
        return_samples=return_count,
    )
    normal1, normal2 = _parallel_transport_frame(centerline)
    embedding = TubeEmbedding(
        centerline=centerline,
        normal1=normal1,
        normal2=normal2,
        track_fraction=track_fraction,
    )

    abstract_dynamics = dynamics or AbstractDynamics()
    abstract_dynamics.track_fraction = track_fraction

    def abstract_field(state: Array) -> Array:
        return abstract_dynamics(np.asarray(state, dtype=float))

    def spatial_field(state: Array) -> Array:
        theta, u, v = embedding.decode(state)
        jacobian = embedding.jacobian(theta, u, v)
        return jacobian @ abstract_field(np.array([theta, u, v], dtype=float))

    steps = int(np.ceil(total_time / dt))
    time, abstract = rk4_integrate(
        abstract_field,
        initial_abstract_state,
        dt=dt,
        steps=steps,
    )
    spatial = np.array([embedding.encode(*state) for state in abstract], dtype=float)

    transient_steps = int(np.floor(transient_time / dt))
    orbit = OrbitRecord(
        time=time[transient_steps:] - time[transient_steps],
        abstract=abstract[transient_steps:],
        spatial=spatial[transient_steps:],
    )
    guided_segment = _find_guided_segment(orbit, curve_samples, track_fraction)

    separation_peak, separation_final = _estimate_separation_growth(
        abstract_field,
        initial_state=initial_abstract_state,
        perturbation=(0.0, 1e-6, 0.0),
        dt=dt,
        steps=max(1, int(np.ceil(18.0 / dt))),
    )

    metadata: dict[str, float | int | str] = {
        "model_type": "custom_guided_tube_flow",
        "track_fraction": float(track_fraction),
        "num_curve_samples": int(num_curve_samples),
        "return_samples": int(return_count),
        "dt": float(dt),
        "total_time": float(total_time),
        "transient_time": float(transient_time),
        "track_height": float(track_height),
        "return_height": float(return_height),
        "separation_peak_ratio": float(separation_peak),
        "separation_final_ratio": float(separation_final),
    }

    return GuidedChaoticField(
        vector_field=spatial_field,
        abstract_field=abstract_field,
        embedding=embedding,
        orbit=orbit,
        guided_segment=guided_segment,
        curve_samples=curve_samples,
        metadata=metadata,
    )


def run_self_checks() -> dict[str, float]:
    def gamma(s: float) -> tuple[float, float]:
        angle = 2.0 * np.pi * s
        return (
            4.5 * np.cos(angle) + 0.7 * np.cos(3.0 * angle),
            2.2 * np.sin(angle) - 0.5 * np.sin(2.0 * angle),
        )

    model = build_guided_chaotic_field(gamma)
    guided = model.guided_segment
    section = model.orbit.abstract[(model.orbit.abstract[:, 0] % 1.0) < 0.01, 1:]
    spread = float(np.std(section[:, 0]) + np.std(section[:, 1])) if len(section) else 0.0
    max_radius = float(np.max(np.linalg.norm(model.orbit.spatial, axis=1)))
    return {
        "max_projection_error": guided.max_projection_error,
        "mean_projection_error": guided.mean_projection_error,
        "section_spread": spread,
        "max_spatial_radius": max_radius,
        "separation_peak_ratio": float(model.metadata["separation_peak_ratio"]),
        "separation_final_ratio": float(model.metadata["separation_final_ratio"]),
    }


def build_demo_model() -> GuidedChaoticField:
    def gamma(s: float) -> tuple[float, float]:
        angle = 2.0 * np.pi * s
        radius = 4.0 + 0.6 * np.cos(3.0 * angle)
        return (
            radius * np.cos(angle),
            0.8 * radius * np.sin(angle),
        )

    return build_guided_chaotic_field(gamma)


if __name__ == "__main__":
    diagnostics = run_self_checks()
    rounded = {key: round(value, 6) for key, value in diagnostics.items()}
    print("self_checks:", rounded)
