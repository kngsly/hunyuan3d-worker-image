"""
Replace hy3dpaint/DifferentiableRenderer/mesh_utils.py entirely to remove the
bpy (Blender) dependency.  The only function the paint pipeline imports from
this module is convert_obj_to_glb, so we replace the whole file with a minimal
trimesh-based implementation.
"""

import pathlib, sys

TARGET = pathlib.Path("hy3dpaint/DifferentiableRenderer/mesh_utils.py")

REPLACEMENT = '''\
"""mesh_utils.py — patched to remove bpy dependency, using trimesh instead."""

def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
) -> bool:
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
'''

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found — repo layout may have changed", file=sys.stderr)
    sys.exit(1)

# Read original to log what we're replacing
original = TARGET.read_text()
has_bpy = "import bpy" in original
has_func = "def convert_obj_to_glb" in original
print(f"patching {TARGET}: has_bpy={has_bpy}, has_convert_obj_to_glb={has_func}")
print(f"original file: {len(original)} bytes, {original.count(chr(10))} lines")

# Overwrite entirely
TARGET.write_text(REPLACEMENT)
print(f"patched: {TARGET} replaced ({len(REPLACEMENT)} bytes)")

# Verify the patch works at import time
sys.path.insert(0, str(TARGET.parent.parent))
sys.path.insert(0, str(TARGET.parent))
try:
    import importlib
    spec = importlib.util.spec_from_file_location("mesh_utils", str(TARGET))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "convert_obj_to_glb"), "convert_obj_to_glb not found after patch"
    print("verified: patched mesh_utils.py imports successfully")
except Exception as e:
    print(f"ERROR: patched mesh_utils.py fails to import: {e}", file=sys.stderr)
    sys.exit(1)
