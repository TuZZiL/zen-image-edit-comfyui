import os
import sys

# ComfyUI's loader imports this folder as a flat module name, so relative imports inside it would
# fail (`ModuleNotFoundError: No module named 'zen_image_edit'`). Put the folder on sys.path and
# import siblings by their flat names — same workaround as in the klein-qwen3-adapter node.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zen_nodes import ZenImage21AdapterLoader, ZenImage21TextEncode, comfy_entrypoint  # noqa: E402

__all__ = ["ZenImage21AdapterLoader", "ZenImage21TextEncode", "comfy_entrypoint"]
