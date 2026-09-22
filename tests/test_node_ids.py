"""Node ids must be distinguishable from the upstream node.

`zen-image-edit-comfyui` registers `ZenImage21AdapterLoader` and
`ZenImage21TextEncode`. ComfyUI keys its node registry on those ids, so two
repos claiming the same key collide: whichever loads last silently replaces the
other in the node menu, and a workflow opens against the wrong implementation.

This fork therefore suffixes its ids, so both copies can live in one ComfyUI.
Setting `ZEN_NODE_SUFFIX=""` restores the upstream ids for anyone running this
copy alone (and for an upstream pull request).
"""
import os
import subprocess
import sys
from pathlib import Path

import zen_nodes

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def _ids():
    return (
        zen_nodes.ZenImage21AdapterLoader.define_schema().node_id,
        zen_nodes.ZenImage21TextEncode.define_schema().node_id,
    )


def test_node_ids_are_suffixed_by_default():
    adapter_id, encode_id = _ids()

    assert adapter_id != "ZenImage21AdapterLoader", (
        "the adapter loader must not claim the upstream node id")
    assert encode_id != "ZenImage21TextEncode", (
        "the text encoder must not claim the upstream node id")
    assert adapter_id.endswith(zen_nodes.NODE_SUFFIX)
    assert encode_id.endswith(zen_nodes.NODE_SUFFIX)


def test_suffix_can_be_cleared_for_a_single_install():
    """ZEN_NODE_SUFFIX="" restores the upstream ids.

    Run in a subprocess on purpose: reloading the module in-process would either
    depend on test ordering or leak a mutated module into sibling tests.
    """
    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(TESTS_DIR)!r});"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "import conftest;"          # installs the comfy stub
        "import zen_nodes;"
        "print(zen_nodes.ZenImage21AdapterLoader.define_schema().node_id);"
        "print(zen_nodes.ZenImage21TextEncode.define_schema().node_id)"
    )
    env = dict(os.environ, ZEN_NODE_SUFFIX="")

    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, env=env, cwd=str(REPO_ROOT))

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines == ["ZenImage21AdapterLoader", "ZenImage21TextEncode"], (
        f"an empty suffix must restore the upstream ids, got {lines}")


def test_display_names_differ_from_upstream():
    adapter = zen_nodes.ZenImage21AdapterLoader.define_schema().display_name
    encode = zen_nodes.ZenImage21TextEncode.define_schema().display_name

    assert zen_nodes.NODE_SUFFIX in adapter
    assert zen_nodes.NODE_SUFFIX in encode


def test_both_nodes_are_suffixed_consistently():
    adapter_id, encode_id = _ids()

    assert adapter_id.endswith(encode_id[len("ZenImage21TextEncode"):]), (
        "both nodes must use the same suffix — a mixed pair would be confusing")
