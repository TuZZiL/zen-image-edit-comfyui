"""The entry point must not inherit another checkout's modules.

ComfyUI imports each custom-node folder as a flat module and puts the folder on
``sys.path``, so every repo's ``zen_nodes``/``fusion_lib`` compete for the same
``sys.modules`` names. With both this fork and the upstream node installed,
whichever loads first owns those names — and the second repo would then export
the *first* repo's classes, silently.

``__init__.py`` therefore loads its siblings under unique names. This test locks
that in: a decoy ``zen_nodes`` is planted in ``sys.modules`` before this repo's
entry point runs, and the exported classes must still be this repo's.
"""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _plant_decoy(tmp_path):
    decoy = tmp_path / "zen_nodes.py"
    decoy.write_text(
        "class ZenImage21AdapterLoader:\n"
        "    pass\n"
        "class ZenImage21TextEncode:\n"
        "    pass\n"
        "def comfy_entrypoint():\n"
        "    pass\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("zen_nodes", decoy)
    module = importlib.util.module_from_spec(spec)
    sys.modules["zen_nodes"] = module
    spec.loader.exec_module(module)
    return module


def _load_entry_point():
    spec = importlib.util.spec_from_file_location(
        "zen_image_edit_entry_probe", REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["zen_image_edit_entry_probe"] = module
    spec.loader.exec_module(module)
    return module


def test_entry_point_ignores_a_foreign_zen_nodes(tmp_path):
    decoy = _plant_decoy(tmp_path)
    try:
        entry = _load_entry_point()

        assert entry.ZenImage21AdapterLoader is not decoy.ZenImage21AdapterLoader, (
            "the entry point exported another checkout's class — dual install would "
            "silently run the wrong implementation")
        assert entry.ZenImage21TextEncode is not decoy.ZenImage21TextEncode
    finally:
        sys.modules.pop("zen_nodes", None)
        sys.modules.pop("zen_image_edit_entry_probe", None)


def test_exported_classes_are_this_repos_own(tmp_path):
    _plant_decoy(tmp_path)
    try:
        entry = _load_entry_point()

        node_id = entry.ZenImage21AdapterLoader.define_schema().node_id

        assert node_id.endswith("Plus"), (
            "the exported loader must be this repo's, so it carries this repo's suffix")
    finally:
        sys.modules.pop("zen_nodes", None)
        sys.modules.pop("zen_image_edit_entry_probe", None)


def test_entry_point_restores_the_foreign_module():
    """Loading this repo must not evict someone else's `fusion_lib` from sys.modules."""
    sentinel = object()
    sys.modules["fusion_lib"] = sentinel  # type: ignore[assignment]
    try:
        _load_entry_point()

        assert sys.modules.get("fusion_lib") is sentinel, (
            "the entry point must leave the other checkout's fusion_lib untouched")
    finally:
        sys.modules.pop("fusion_lib", None)
        sys.modules.pop("zen_image_edit_entry_probe", None)
