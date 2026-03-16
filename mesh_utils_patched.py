"""mesh_utils.py — patched to remove bpy dependency.

Provides load_mesh, save_mesh, save_obj_mesh, and convert_obj_to_glb
without requiring Blender (bpy).
"""

import os
import cv2
import numpy as np
from io import StringIO
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_extract_attribute(obj, attr_path, default=None):
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default


def _convert_to_numpy(data, dtype):
    if data is None:
        return None
    return np.asarray(data, dtype=dtype)


def _get_base_path_and_name(mesh_path):
    base_path = os.path.splitext(mesh_path)[0]
    name = os.path.basename(base_path)
    return base_path, name


def _save_texture_map(texture, base_path, suffix="", image_format=".jpg", color_convert=None):
    path = f"{base_path}{suffix}{image_format}"
    processed_texture = (texture * 255).astype(np.uint8)
    if color_convert is not None:
        processed_texture = cv2.cvtColor(processed_texture, color_convert)
        cv2.imwrite(path, processed_texture)
    else:
        cv2.imwrite(path, processed_texture[..., ::-1])  # RGB to BGR
    return os.path.basename(path)


def _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name):
    buffer = StringIO()
    buffer.write(f"mtllib {name}.mtl\n")
    buffer.write(f"o {name}\n")
    np.savetxt(buffer, vtx_pos, fmt="v %.6f %.6f %.6f")
    np.savetxt(buffer, vtx_uv, fmt="vt %.6f %.6f")
    buffer.write("s 0\n")
    buffer.write("usemtl Material\n")
    pos_idx_plus1 = pos_idx + 1
    uv_idx_plus1 = uv_idx + 1
    face_format = np.frompyfunc(lambda *x: f"{int(x[0])}/{int(x[1])}", 2, 1)
    faces = face_format(pos_idx_plus1, uv_idx_plus1)
    face_strings = [f"f {' '.join(face)}" for face in faces]
    buffer.write("\n".join(face_strings) + "\n")
    return buffer.getvalue()


def _write_mtl_properties(f, properties):
    for key, value in properties.items():
        if isinstance(value, (list, tuple)):
            f.write(f"{key} {' '.join(map(str, value))}\n")
        else:
            f.write(f"{key} {value}\n")


def _create_mtl_file(base_path, texture_maps, is_pbr):
    mtl_path = f"{base_path}.mtl"
    with open(mtl_path, "w") as f:
        f.write("newmtl Material\n")
        if is_pbr:
            properties = {
                "Kd": [0.800, 0.800, 0.800],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 2,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)
            map_configs = [("metallic", "map_Pm"), ("roughness", "map_Pr"), ("normal", "map_Bump -bm 1.0")]
            for texture_key, mtl_key in map_configs:
                if texture_key in texture_maps:
                    f.write(f"{mtl_key} {texture_maps[texture_key]}\n")
        else:
            properties = {
                "Ns": 250.000000,
                "Ka": [0.200, 0.200, 0.200],
                "Kd": [0.800, 0.800, 0.800],
                "Ks": [0.500, 0.500, 0.500],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 3,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mesh(mesh):
    """Load mesh data including vertices, faces, UV coordinates and texture."""
    vtx_pos = _safe_extract_attribute(mesh, "vertices")
    pos_idx = _safe_extract_attribute(mesh, "faces")
    vtx_uv = _safe_extract_attribute(mesh, "visual.uv")
    uv_idx = pos_idx
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)
    texture_data = None
    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data


def save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture,
                  metallic=None, roughness=None, normal=None):
    """Save mesh as OBJ file with textures and material."""
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)
    base_path, name = _get_base_path_and_name(mesh_path)
    obj_content = _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name)
    with open(mesh_path, "w") as obj_file:
        obj_file.write(obj_content)
    texture_maps = {}
    texture_maps["diffuse"] = _save_texture_map(texture, base_path)
    if metallic is not None:
        texture_maps["metallic"] = _save_texture_map(
            metallic, base_path, "_metallic", color_convert=cv2.COLOR_RGB2GRAY)
    if roughness is not None:
        texture_maps["roughness"] = _save_texture_map(
            roughness, base_path, "_roughness", color_convert=cv2.COLOR_RGB2GRAY)
    if normal is not None:
        texture_maps["normal"] = _save_texture_map(normal, base_path, "_normal")
    _create_mtl_file(base_path, texture_maps, metallic is not None)


def save_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture,
              metallic=None, roughness=None, normal=None):
    """Save mesh using OBJ format."""
    save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture,
                  metallic=metallic, roughness=roughness, normal=normal)


def convert_obj_to_glb(obj_path, glb_path, shade_type="SMOOTH",
                       auto_smooth_angle=60, merge_vertices=False):
    """Convert OBJ file to GLB format using trimesh (bpy-free)."""
    try:
        import trimesh
        scene = trimesh.load(obj_path, process=False)
        scene.export(glb_path, file_type="glb")
        return True
    except Exception as e:
        print(f"[mesh_utils] convert_obj_to_glb failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False
