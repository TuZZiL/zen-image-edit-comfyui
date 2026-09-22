"""ComfyUI entry point.

ComfyUI imports every custom node folder as a flat module and puts the folder on
``sys.path``, so sibling modules are imported by bare name. That works for a
single checkout, but it breaks when two checkouts of this node are installed at
once (this fork next to the upstream `zen-image-edit-comfyui`): both ship a
``zen_nodes.py`` and a ``fusion_lib.py``, and Python caches modules by name in
``sys.modules``. Whichever repo loads first wins, and the second one silently
re-exports the *first* repo's classes — the other implementation's node ids,
the other implementation's behaviour.

So this entry point does not use a bare ``from zen_nodes import ...``. It loads
both sibling modules under unique, repo-specific names instead, and temporarily
points ``fusion_lib`` at this repo's copy while ``zen_nodes`` executes (because
``zen_nodes`` imports it flat). Nothing about the node code itself changes.
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# Unique per repo: ComfyUI puts every custom-node folder on sys.path, so two
# checkouts must not share a module name.
_MODULE_PREFIX = "zen_image_edit_tuzzil_"


def _load_module(unique_name, filename):
    """Load a sibling .py file under `unique_name`, bypassing sys.modules by name."""
    if unique_name in sys.modules:
        del sys.modules[unique_name]
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename} from {_HERE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique_name, None)
        raise
    return module


# fusion_lib must be importable under its flat name *while* zen_nodes executes,
# otherwise `import fusion_lib` inside zen_nodes binds to the other checkout.
_own_fusion = _load_module(_MODULE_PREFIX + "fusion_lib", "fusion_lib.py")

_previous_fusion = sys.modules.get("fusion_lib")
sys.modules["fusion_lib"] = _own_fusion
try:
    _zen_nodes = _load_module(_MODULE_PREFIX + "zen_nodes", "zen_nodes.py")
finally:
    # Leave the other checkout's fusion_lib (if any) exactly as we found it.
    if _previous_fusion is not None:
        sys.modules["fusion_lib"] = _previous_fusion
    else:
        sys.modules.pop("fusion_lib", None)

ZenImage21AdapterLoader = _zen_nodes.ZenImage21AdapterLoader
ZenImage21TextEncode = _zen_nodes.ZenImage21TextEncode
comfy_entrypoint = _zen_nodes.comfy_entrypoint

__all__ = ["ZenImage21AdapterLoader", "ZenImage21TextEncode", "comfy_entrypoint"]
