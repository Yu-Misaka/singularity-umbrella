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
CAMERA_NAME = "GuidedChaoticCamera"
FOLLOWER_RADIUS = 0.035
FOLLOWER_GROUP_PREFIX = "GuidedChaoticFollowerGN"


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


def _clear_generated_node_groups() -> None:
    for node_group in list(bpy.data.node_groups):
        if node_group.name.startswith(FOLLOWER_GROUP_PREFIX):
            bpy.data.node_groups.remove(node_group, do_unlink=True)


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


def _set_helper_curve_display(obj, *, rgba: tuple[float, float, float, float]) -> None:
    obj.hide_render = True
    obj.display_type = "WIRE"
    obj.color = rgba
    if hasattr(obj.data, "bevel_depth"):
        obj.data.bevel_depth = 0.0
    if hasattr(obj.data, "bevel_resolution"):
        obj.data.bevel_resolution = 0


def _ensure_projection_camera(paths_payload: dict):
    """Create or update a top-down orthographic camera matching the projection model."""

    world = paths_payload["world"]
    canvas_extent = float(world["canvas_extent"])
    canvas_z = float(world["canvas_z"])
    local_depth_scale = float(world["local_to_world_depth_scale"])

    camera_object = bpy.data.objects.get(CAMERA_NAME)
    if camera_object is None or camera_object.type != "CAMERA":
        camera_data = bpy.data.cameras.new(CAMERA_NAME)
        camera_object = bpy.data.objects.new(CAMERA_NAME, camera_data)
        bpy.context.scene.collection.objects.link(camera_object)

    camera_object.data.type = "ORTHO"
    camera_object.data.ortho_scale = canvas_extent * 1.05
    camera_object.location = (0.0, 0.0, canvas_z + max(canvas_extent, local_depth_scale * 8.0, 8.0))
    camera_object.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = camera_object
    return camera_object


def _make_poly_curve(name: str, points: list[list[float]], *, collection):
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = 0.0
    curve.bevel_resolution = 0
    curve.use_path = True
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for spline_point, point in zip(spline.points, points, strict=False):
        spline_point.co = (point[0], point[1], point[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    return obj


def _make_follower_host(name: str, *, collection):
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


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


def _set_viewport_only_visibility(obj, *, hide_before: int, hide_after: int) -> None:
    obj.hide_render = True
    for frame, hidden in [
        (hide_before, True),
        (hide_before + 1, False),
        (hide_after, False),
        (hide_after + 1, True),
    ]:
        obj.hide_viewport = hidden
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)


def _key_custom_property(obj, key: str, frame_values: list[tuple[int, float]]) -> None:
    for frame, value in frame_values:
        obj[key] = float(value)
        obj.keyframe_insert(data_path=f'["{key}"]', frame=int(frame))


def _ensure_group_socket(node_group, *, name: str, in_out: str, socket_type: str) -> None:
    interface = getattr(node_group, "interface", None)
    if interface is not None and hasattr(interface, "new_socket"):
        interface.new_socket(name=name, in_out=in_out, socket_type=socket_type)
        return
    if in_out == "INPUT":
        node_group.inputs.new(socket_type, name)
    else:
        node_group.outputs.new(socket_type, name)


def _build_follower_geometry_nodes(
    follower_obj,
    *,
    orbit_obj,
    part_schedule: dict,
    radius: float,
    material,
    orbit_sample_count: int,
):
    node_group = bpy.data.node_groups.new(
        name=f"{FOLLOWER_GROUP_PREFIX}_{follower_obj.name}",
        type="GeometryNodeTree",
    )
    _ensure_group_socket(node_group, name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    _ensure_group_socket(node_group, name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = node_group.nodes
    links = node_group.links
    nodes.clear()

    group_input = nodes.new(type="NodeGroupInput")
    group_output = nodes.new(type="NodeGroupOutput")
    object_info = nodes.new(type="GeometryNodeObjectInfo")
    resample_curve = nodes.new(type="GeometryNodeResampleCurve")
    scene_time = nodes.new(type="GeometryNodeInputSceneTime")
    subtract = nodes.new(type="ShaderNodeMath")
    divide = nodes.new(type="ShaderNodeMath")
    sample_curve = nodes.new(type="GeometryNodeSampleCurve")
    mesh_line = nodes.new(type="GeometryNodeMeshLine")
    set_position = nodes.new(type="GeometryNodeSetPosition")
    ico_sphere = nodes.new(type="GeometryNodeMeshIcoSphere")
    instance_on_points = nodes.new(type="GeometryNodeInstanceOnPoints")
    realize_instances = nodes.new(type="GeometryNodeRealizeInstances")
    set_material = nodes.new(type="GeometryNodeSetMaterial")

    group_input.location = (-1240.0, 0.0)
    object_info.location = (-1040.0, 240.0)
    resample_curve.location = (-820.0, 240.0)
    scene_time.location = (-1040.0, -220.0)
    subtract.location = (-820.0, -220.0)
    divide.location = (-620.0, -220.0)
    sample_curve.location = (-400.0, 180.0)
    mesh_line.location = (-400.0, -80.0)
    set_position.location = (-180.0, -80.0)
    ico_sphere.location = (-180.0, -320.0)
    instance_on_points.location = (40.0, -80.0)
    realize_instances.location = (260.0, -80.0)
    set_material.location = (480.0, -80.0)
    group_output.location = (720.0, -80.0)

    object_info.inputs["Object"].default_value = orbit_obj

    if hasattr(resample_curve, "mode"):
        resample_curve.mode = "COUNT"
    resample_curve.inputs["Count"].default_value = int(max(2, orbit_sample_count))

    subtract.operation = "SUBTRACT"
    subtract.inputs[1].default_value = float(part_schedule["start_frame"])

    divide.operation = "DIVIDE"
    divide.use_clamp = True
    divide.inputs[1].default_value = float(max(1, part_schedule["travel_frames"]))

    if hasattr(sample_curve, "mode"):
        sample_curve.mode = "FACTOR"

    mesh_line.inputs["Count"].default_value = 1
    ico_sphere.inputs["Radius"].default_value = float(radius)
    if "Subdivisions" in ico_sphere.inputs:
        ico_sphere.inputs["Subdivisions"].default_value = 2
    set_material.inputs["Material"].default_value = material

    links.new(object_info.outputs["Geometry"], resample_curve.inputs["Curve"])
    links.new(resample_curve.outputs["Curve"], sample_curve.inputs["Curves"])
    links.new(scene_time.outputs["Frame"], subtract.inputs[0])
    links.new(subtract.outputs["Value"], divide.inputs[0])
    links.new(divide.outputs["Value"], sample_curve.inputs["Factor"])
    links.new(mesh_line.outputs["Mesh"], set_position.inputs["Geometry"])
    links.new(sample_curve.outputs["Position"], set_position.inputs["Position"])
    links.new(set_position.outputs["Geometry"], instance_on_points.inputs["Points"])
    links.new(ico_sphere.outputs["Mesh"], instance_on_points.inputs["Instance"])
    links.new(instance_on_points.outputs["Instances"], realize_instances.inputs["Geometry"])
    links.new(realize_instances.outputs["Geometry"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])

    modifier = follower_obj.modifiers.new(name="FollowerGeometry", type="NODES")
    modifier.node_group = node_group
    return modifier


def _animate_follower(follower, orbit_obj, part_schedule: dict, *, material, orbit_sample_count: int) -> None:
    _build_follower_geometry_nodes(
        follower,
        orbit_obj=orbit_obj,
        part_schedule=part_schedule,
        radius=FOLLOWER_RADIUS,
        material=material,
        orbit_sample_count=orbit_sample_count,
    )

    follower["enter_target_frame"] = int(part_schedule["enter_target_frame"])
    follower["exit_target_frame"] = int(part_schedule["exit_target_frame"])
    follower["entry_progress"] = float(part_schedule["entry_progress"])
    follower["exit_progress"] = float(part_schedule["exit_progress"])

    _key_custom_property(
        follower,
        "path_progress",
        [
            (part_schedule["start_frame"], 0.0),
            (part_schedule["enter_target_frame"], float(part_schedule["entry_progress"])),
            (part_schedule["exit_target_frame"], float(part_schedule["exit_progress"])),
            (part_schedule["end_frame"], 1.0),
        ],
    )
    _key_custom_property(
        follower,
        "enter_target_event",
        [
            (max(0, part_schedule["enter_target_frame"] - 1), 0.0),
            (part_schedule["enter_target_frame"], 1.0),
            (part_schedule["enter_target_frame"] + 1, 0.0),
        ],
    )
    _key_custom_property(
        follower,
        "inside_target_window",
        [
            (max(0, part_schedule["enter_target_frame"] - 1), 0.0),
            (part_schedule["enter_target_frame"], 1.0),
            (part_schedule["exit_target_frame"], 1.0),
            (part_schedule["exit_target_frame"] + 1, 0.0),
        ],
    )
    _key_custom_property(
        follower,
        "exit_target_event",
        [
            (max(0, part_schedule["exit_target_frame"] - 1), 0.0),
            (part_schedule["exit_target_frame"], 1.0),
            (part_schedule["exit_target_frame"] + 1, 0.0),
        ],
    )


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
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)


def build_scene(paths_path: str | Path = DEFAULT_PATHS, schedule_path: str | Path = DEFAULT_SCHEDULE) -> None:
    """Import the exported paths and schedule into the current Blender scene."""

    _ensure_blender()
    paths_payload = _load_json(paths_path)
    schedule_payload = _load_json(schedule_path)
    schedule_lookup = {item["part_id"]: item for item in schedule_payload["parts"]}

    collection = _ensure_collection(COLLECTION_NAME)
    _clear_collection(collection)
    _clear_generated_node_groups()

    follower_material = _make_material("FollowerMaterial", rgba=(0.95, 0.96, 0.99, 1.0))

    scene = bpy.context.scene
    scene.render.fps = int(schedule_payload["fps"])
    scene.frame_start = int(schedule_payload["scene_frame_start"])
    scene.frame_end = int(schedule_payload["scene_frame_end"])
    _ensure_projection_camera(paths_payload)

    for part in paths_payload["parts"]:
        schedule = schedule_lookup.get(part["part_id"])
        if schedule is None:
            continue

        orbit_obj = _make_poly_curve(
            part["orbit_object_name"],
            part["orbit_points_world"],
            collection=collection,
        )
        guided_obj = _make_poly_curve(
            part["guided_object_name"],
            part["guided_points_world"],
            collection=collection,
        )
        target_obj = _make_poly_curve(
            f"{part['part_id']}_target",
            part["target_points_world"],
            collection=collection,
        )
        follower_obj = _make_follower_host(
            part["follower_object_name"],
            collection=collection,
        )

        _set_helper_curve_display(orbit_obj, rgba=(0.40, 0.66, 0.88, 0.35))
        _set_helper_curve_display(guided_obj, rgba=(0.12, 0.46, 0.86, 1.0))
        _set_helper_curve_display(target_obj, rgba=(0.15, 0.16, 0.18, 0.70))

        _set_viewport_only_visibility(orbit_obj, hide_before=schedule["hide_before_frame"], hide_after=schedule["hide_after_frame"])
        _set_viewport_only_visibility(target_obj, hide_before=schedule["hide_before_frame"], hide_after=schedule["hide_after_frame"])
        _set_visibility(follower_obj, hide_before=schedule["hide_before_frame"], hide_after=schedule["hide_after_frame"])
        _animate_guided_visibility(guided_obj, schedule)
        _animate_follower(
            follower_obj,
            orbit_obj,
            schedule,
            material=follower_material,
            orbit_sample_count=len(part["orbit_points_world"]),
        )


if __name__ == "__main__":  # pragma: no cover - Blender-only runtime
    build_scene()
