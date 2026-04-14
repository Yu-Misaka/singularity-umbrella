from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core import AbstractDynamics, GuidedChaoticField, build_guided_chaotic_field
from preprocess import load_curve_fit


def _system_payload(
    model: GuidedChaoticField,
    *,
    source: dict[str, Any],
    model_kwargs: dict[str, Any],
    dynamics: AbstractDynamics,
    initial_abstract_state: tuple[float, float, float],
) -> dict[str, Any]:
    """Serialize the ingredients needed to reconstruct one guided chaotic field."""

    return {
        "source": source,
        "equations": {
            "abstract_coordinates": ["theta", "u", "v"],
            "abstract_system": [
                "dtheta/dt = omega",
                (
                    "du/dt = -(global_u_damping + track_rate * track_weight(theta) + "
                    "pretrack_rate * pretrack_weight(theta)) * u + drift_gain * drift_weight(theta) * v "
                    "+ stretch_gain * stretch_weight(theta) * u"
                ),
                (
                    "dv/dt = -(global_v_damping + track_rate * track_weight(theta) + "
                    "pretrack_rate * pretrack_weight(theta)) * v + kick_gain * kick_weight(theta) "
                    "* sin(nonlinearity * u)"
                ),
                "X(theta,u,v) = centerline(theta) + u * normal1(theta) + v * normal2(theta)",
                "dX/dt = J_X(theta,u,v) @ [dtheta/dt, du/dt, dv/dt]^T",
            ],
        },
        "model_kwargs": model_kwargs,
        "initial_abstract_state": list(initial_abstract_state),
        "dynamics": asdict(dynamics),
        "guided_segment": asdict(model.guided_segment),
        "metadata": model.metadata,
        "curve_samples": model.curve_samples.tolist(),
        "embedding": {
            "centerline": model.embedding.centerline.tolist(),
            "normal1": model.embedding.normal1.tolist(),
            "normal2": model.embedding.normal2.tolist(),
            "track_fraction": float(model.embedding.track_fraction),
        },
    }


def export_system_from_curve_fit(
    curve_fit_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_kwargs: dict[str, Any] | None = None,
    dynamics_kwargs: dict[str, Any] | None = None,
    initial_abstract_state: tuple[float, float, float] = (0.0, 0.03, -0.02),
    source_extra: dict[str, Any] | None = None,
) -> Path:
    """Build a model from one `CurveFit` file and export a reconstructible system definition."""

    curve_fit_path = Path(curve_fit_path)
    fit = load_curve_fit(curve_fit_path)
    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    dynamics = AbstractDynamics(**({} if dynamics_kwargs is None else dict(dynamics_kwargs)))
    model = build_guided_chaotic_field(
        fit.as_array(),
        num_curve_samples=fit.num_samples,
        dynamics=dynamics,
        initial_abstract_state=initial_abstract_state,
        **resolved_model_kwargs,
    )

    source = {
        "curve_fit_path": str(curve_fit_path),
        "image_path": fit.image_path,
        "image_size": list(fit.image_size),
        "bounding_box": list(fit.bounding_box),
        "num_samples": int(fit.num_samples),
        "target_extent": float(fit.target_extent),
    }
    if source_extra:
        source.update(source_extra)

    payload = _system_payload(
        model,
        source=source,
        model_kwargs=resolved_model_kwargs,
        dynamics=dynamics,
        initial_abstract_state=initial_abstract_state,
    )

    output = Path(output_path) if output_path is not None else curve_fit_path.with_name(f"{curve_fit_path.stem}_system.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def export_systems_from_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    limit_parts: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    dynamics_kwargs: dict[str, Any] | None = None,
    initial_abstract_state: tuple[float, float, float] = (0.0, 0.03, -0.02),
) -> dict[str, Any]:
    """Export one system-definition JSON per accepted part listed in a dissection manifest."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = list(manifest["parts"])
    if limit_parts is not None:
        parts = parts[:limit_parts]

    destination = Path(output_dir) if output_dir is not None else Path(manifest["output_dir"]) / "systems"
    destination.mkdir(parents=True, exist_ok=True)

    system_records: list[dict[str, Any]] = []
    for part in parts:
        output_path = destination / f"{part['part_id']}_system.json"
        exported = export_system_from_curve_fit(
            part["curve_fit_path"],
            output_path=output_path,
            model_kwargs=model_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            initial_abstract_state=initial_abstract_state,
            source_extra={
                "part_id": part["part_id"],
                "part_svg_path": part["svg_path"],
                "source_subpath_index": part["source_subpath_index"],
                "split_index": part["split_index"],
                "normalized_center": part["normalized_center"],
                "fit_scale": part["fit_scale"],
            },
        )
        system_records.append(
            {
                "part_id": part["part_id"],
                "system_path": str(exported),
                "curve_fit_path": part["curve_fit_path"],
                "svg_path": part["svg_path"],
            }
        )

    index_payload = {
        "manifest_path": str(manifest_path),
        "output_dir": str(destination),
        "exported_part_count": len(system_records),
        "model_kwargs": {} if model_kwargs is None else dict(model_kwargs),
        "dynamics_kwargs": {} if dynamics_kwargs is None else dict(dynamics_kwargs),
        "initial_abstract_state": list(initial_abstract_state),
        "parts": system_records,
    }
    index_path = destination / "systems_index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    index_payload["index_path"] = str(index_path)
    return index_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export reconstructible guided-chaotic system definitions.")
    parser.add_argument("input_path", help="Path to a CurveFit JSON or a dissection manifest JSON")
    parser.add_argument("--from-manifest", action="store_true", help="Interpret input_path as a dissect_svg manifest")
    parser.add_argument("--output", default=None, help="Output JSON path for single-curve export")
    parser.add_argument("--output-dir", default=None, help="Output directory for manifest export")
    parser.add_argument("--limit-parts", type=int, default=None, help="Optional cap for manifest export")
    parser.add_argument("--total-time", type=float, default=35.0, help="Simulation time passed to core.py")
    parser.add_argument("--transient-time", type=float, default=8.0, help="Transient time discarded before analysis")
    parser.add_argument("--dt", type=float, default=0.005, help="RK4 step size")
    parser.add_argument("--track-height", type=float, default=2.0, help="Guided-segment centerline height")
    parser.add_argument("--return-height", type=float, default=4.0, help="Return-segment centerline height")
    return parser


def main() -> None:
    """CLI entry point for system export."""

    args = _build_parser().parse_args()
    model_kwargs = {
        "total_time": float(args.total_time),
        "transient_time": float(args.transient_time),
        "dt": float(args.dt),
        "track_height": float(args.track_height),
        "return_height": float(args.return_height),
    }

    if args.from_manifest:
        result = export_systems_from_manifest(
            args.input_path,
            output_dir=args.output_dir,
            limit_parts=args.limit_parts,
            model_kwargs=model_kwargs,
        )
        print(
            json.dumps(
                {
                    "index_path": result["index_path"],
                    "exported_part_count": result["exported_part_count"],
                    "output_dir": result["output_dir"],
                },
                ensure_ascii=True,
            )
        )
        return

    output = export_system_from_curve_fit(
        args.input_path,
        output_path=args.output,
        model_kwargs=model_kwargs,
    )
    print(json.dumps({"output_path": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
