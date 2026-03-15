"""
Patch hy3dpaint/DifferentiableRenderer/mesh_utils.py to remove the bpy
(Blender) dependency and replace convert_obj_to_glb with a trimesh-based
implementation.

bpy is huge and unnecessary — the only thing the worker needs is a simple
OBJ → GLB conversion, which trimesh handles fine.
"""

import pathlib, re

TARGET = pathlib.Path("hy3dpaint/DifferentiableRenderer/mesh_utils.py")

REPLACEMENT = '''\
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
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        mesh.export(glb_path)
        return True
    except Exception:
        return False
'''

code = TARGET.read_text()

# Comment out `import bpy`
code = re.sub(r'^import bpy\b', '# import bpy  # patched: using trimesh', code, flags=re.MULTILINE)

# Replace the convert_obj_to_glb function (and any bpy helpers it calls)
# by finding it and replacing up to the next top-level def/class/EOF
pattern = r'def convert_obj_to_glb\(.*?\n(?=(?:def |class )|\Z)'
if re.search(pattern, code, flags=re.DOTALL):
    code = re.sub(pattern, REPLACEMENT + '\n', code, flags=re.DOTALL)
    print("patched: replaced convert_obj_to_glb with trimesh version")
else:
    # Fallback: just append the function at the end
    code += '\n' + REPLACEMENT
    print("patched: appended trimesh convert_obj_to_glb (original not found)")

TARGET.write_text(code)
print(f"patched: {TARGET}")
