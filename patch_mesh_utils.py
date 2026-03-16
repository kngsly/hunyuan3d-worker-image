"""
Replace hy3dpaint/DifferentiableRenderer/mesh_utils.py with our bpy-free version.
"""
import shutil, pathlib, sys, importlib.util

TARGET = pathlib.Path("hy3dpaint/DifferentiableRenderer/mesh_utils.py")
SOURCE = pathlib.Path("/tmp/mesh_utils_patched.py")

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found", file=sys.stderr)
    sys.exit(1)

original = TARGET.read_text()
print(f"patching {TARGET}: original={len(original)} bytes")

shutil.copy2(str(SOURCE), str(TARGET))
print(f"patched: copied {SOURCE} -> {TARGET} ({TARGET.stat().st_size} bytes)")

# Verify all required functions exist
spec = importlib.util.spec_from_file_location("mesh_utils", str(TARGET))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
required = ["load_mesh", "save_mesh", "save_obj_mesh", "convert_obj_to_glb"]
for fn in required:
    assert hasattr(mod, fn), f"{fn} not found after patch"
print(f"verified: {', '.join(required)}")
