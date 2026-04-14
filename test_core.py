import unittest

import numpy as np

from core import build_guided_chaotic_field


def sample_gamma(s: float) -> tuple[float, float]:
    angle = 2.0 * np.pi * s
    return (
        4.0 * np.cos(angle) + 0.8 * np.cos(2.0 * angle),
        2.0 * np.sin(angle) - 0.6 * np.sin(3.0 * angle),
    )


class GuidedFieldTests(unittest.TestCase):
    """Numerical sanity checks for the custom guided chaotic flow."""

    def test_guided_segment_tracks_target_curve(self) -> None:
        model = build_guided_chaotic_field(
            sample_gamma,
            num_curve_samples=140,
            total_time=35.0,
            transient_time=8.0,
            dt=0.005,
        )
        self.assertLess(model.guided_segment.max_projection_error, 0.05)
        self.assertLess(model.guided_segment.mean_projection_error, 0.02)

    def test_spatial_field_stays_bounded_and_finite(self) -> None:
        model = build_guided_chaotic_field(
            sample_gamma,
            num_curve_samples=140,
            total_time=35.0,
            transient_time=8.0,
            dt=0.005,
        )
        orbit = model.simulate(6.0, 0.005, initial_abstract_state=model.orbit.abstract[0])
        self.assertTrue(np.isfinite(orbit.spatial).all())
        self.assertLess(np.max(np.linalg.norm(orbit.spatial, axis=1)), 25.0)

    def test_return_dynamics_show_nontrivial_section_spread(self) -> None:
        model = build_guided_chaotic_field(
            sample_gamma,
            num_curve_samples=140,
            total_time=35.0,
            transient_time=8.0,
            dt=0.005,
        )
        theta = model.orbit.abstract[:, 0] % 1.0
        section = model.orbit.abstract[theta < 0.01, 1:]
        self.assertGreater(len(section), 12)
        spread = float(np.std(section[:, 0]) + np.std(section[:, 1]))
        self.assertGreater(spread, 1.0)
        self.assertGreater(float(model.metadata["separation_peak_ratio"]), 1_000.0)


if __name__ == "__main__":
    unittest.main()
