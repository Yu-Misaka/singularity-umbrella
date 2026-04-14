from __future__ import annotations

import json
from pathlib import Path


try:
    import bpy
except ImportError:  # pragma: no cover - Blender-only runtime
    bpy = None


DEFAULT_PATHS = "experiment_outputs/canary_1/blender_paths.json"
DEFAULT_SCHEDULE = "experiment_outputs/canary_1/animation_schedule.json"
COLLECTION_NAME = "GuidedChaoticPortrait"


def _ensure_blender() -> None:
    if bpy is None:  # pragma: no cover - Blender-only runtime
        raise RuntimeError("blender_import.py must be run from Blender's Python environment")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ensure_collection(name: str):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _clear_collection(collection) -> None:
    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _make_material(name: str, *, rgba: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new(type="ShaderNodeOutputMaterial")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    if "Base Color" in principled.inputs:
        principled.inputs["Base Color"].default_value = rgba
    if "Emission Color" in principled.inputs:
        principled.inputs["Emission Color"].default_value = rgba
    elif "Emission" in principled.inputs:
        principled.inputs["Emission"].default_value = rgba
    if "Emission Strength" in principled.inputs:
        principled.inputs["Emission Strength"].default_value = 0.1
    if "Alpha" in principled.inputs:
        principled.inputs["Alpha"].default_value = rgba[3]
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"
    if hasattr(material, "shadow_method"):
        material.shadow_method = "NONE"
    return material


def _make_poly_curve(name: str, points: list[list[float]], *, bevel_depth: float, collection, material):
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    curve.use_path = True
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for spline_point, point in zip(spline.points, points, strict=False):
        spline_point.co = (point[0], point[1], point[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


def _make_follower(name: str, *, radius: float, collection, material):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0.0, 0.0, 0.0))
    sphere = bpy.context.active_object
    sphere.name = name
    if sphere.data.materials:
        sphere.data.materials[0] = material
    else:
        sphere.data.materials.append(material)
    collection.objects.link(sphere)
    scene_collection = bpy.context.scene.collection
    if sphere.name in scene_collection.objects:
        scene_collection.objects.unlink(sphere)
    return sphere


def _set_visibility(obj, *, hide_before: int, hide_after: int) -> None:
    for frame, hidden in [
        (hide_before, True),
        (hide_before + 1, False),
        (hide_after, False),
        (hide_after + 1, True),
    ]:
        obj.hide_viewport = hidden
        obj.hide_render = hidden
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)


def _animate_follower(follower, curve_obj, part_schedule: dict) -> None:
    constraint = follower.constraints.new(type="FOLLOW_PATH")
    constraint.target = curve_obj
    constraint.use_curve_follow = True
    constraint.use_fixed_location = True
    constraint.forward_axis = "FORWARD_Y"
    constraint.up_axis = "UP_Z"
    curve_obj.data.path_duration = max(1, int(part_schedule["travel_frames"]))
    constraint.offset_factor = 0.0
    constraint.keyframe_insert(data_path="offset_factor", frame=part_schedule["start_frame"])
    constraint.offset_factor = 1.0
    constraint.keyframe_insert(data_path="offset_factor", frame=part_schedule["end_frame"])


def _animate_guided_visibility(obj, part_schedule: dict) -> None:
    reveal_start = part_schedule["guided_reveal_start_frame"]
    reveal_end = part_schedule["guided_reveal_end_frame"]
    for frame, hidden in [
        (part_schedule["hide_before_frame"], True),
        (reveal_start, False),
        (reveal_end, False),
        (reveal_end + 1, True),
    ]:
        obj.hide_viewport = hidden
        obj.hide_render = hidden
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)


def build_scene(paths_path: str | Path = DEFAULT_PATHS, schedule_path: str | Path = DEFAULT_SCHEDULE) -> None:
    """Import the exported paths and schedule into the current Blender scene."""

    _ensure_blender()
    paths_payload = _load_json(paths_path)
    schedule_payload = _load_json(schedule_path)
    schedule_lookup = {item["part_id"]: item for item in schedule_payload["parts"]}

    collection = _ensure_collection(COLLECTION_NAME)
    _clear_collection(collection)

    orbit_material = _make_material("OrbitMaterial", rgba=(0.40, 0.66, 0.88, 0.14))
    guided_material = _make_material("GuidedMaterial", rgba=(0.12, 0.46, 0.86, 0.92))
    target_material = _make_material("TargetMaterial", rgba=(0.15, 0.16, 0.18, 0.70))
    follower_material = _make_material("FollowerMaterial", rgba=(0.95, 0.96, 0.99, 1.0))

    scene = bpy.context.scene
    scene.render.fps = int(schedule_payload["fps"])
    scene.frame_start = int(schedule_payload["scene_frame_start"])
    scene.frame_end = int(schedule_payload["scene_frame_end"])

    for part in paths_payload["parts"]:
        schedule = schedule_lookup.get(part["part_id"])
        if schedule is None:
            continue

        orbit_obj = _make_poly_curve(
            part["orbit_object_name"],
            part["orbit_points_world"],
            bevel_depth=0.010,
            collection=collection,
            material=orbit_material,
        )
        guided_obj = _make_poly_curve(
            part["guided_object_name"],
            part["guided_points_world"],
            bevel_depth=0.020,
            collection=collection,
            material=guided_material,
        )
        target_obj = _make_poly_curve(
            f"{part['part_id']}_target",
            part["target_points_world"],
            bevel_depth=0.006,
            collection=collection,
            material=target_material,
        )
        follower_obj = _make_follower(
            part["follower_object_name"],
            radius=0.035,
            collection=collection,
            material=follower_material,
        )

        _set_visibility(orbit_obj, hide_before=schedule["hide_before_frame"], hide_after=schedule["hide_after_frame"])
        _set_visibility(target_obj, hide_before=schedule["hide_before_frame"], hide_after=schedule["hide_after_frame"])
        _set_visibility(follower_obj, hide_before=schedule["hide_before_frame"], hide_after=schedule["hide_after_frame"])
        _animate_guided_visibility(guided_obj, schedule)
        _animate_follower(follower_obj, orbit_obj, schedule)


if __name__ == "__main__":  # pragma: no cover - Blender-only runtime
    build_scene()
