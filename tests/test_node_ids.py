"""Node ids must be distinguishable from the upstream node.

`zen-image-edit-comfyui` registers `ZenImage21AdapterLoader` and
`ZenImage21TextEncode`. ComfyUI keys its node registry on those ids, so two
repos claiming the same key collide: whichever loads last silently replaces the
other in the node menu, and a workflow opens against the wrong implementation.

This fork therefore suffixes its ids, so both copies can live in one ComfyUI.
Setting `ZEN_NODE_SUFFIX=""` restores the upstream ids for anyone running this
copy alone (and for an upstream pull request).
"""
import importlib

import zen_nodes


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


def test_suffix_can_be_cleared_for_a_single_install(monkeypatch):
    monkeypatch.setenv("ZEN_NODE_SUFFIX", "")
    importlib.reload(zen_nodes)
    try:
        adapter_id, encode_id = _ids()

        assert adapter_id == "ZenImage21AdapterLoader"
        assert encode_id == "ZenImage21TextEncode"
    finally:
        monkeypatch.undo()
        importlib.reload(zen_nodes)


def test_display_names_differ_from_upstream():
    adapter = zen_nodes.ZenImage21AdapterLoader.define_schema().display_name
    encode = zen_nodes.ZenImage21TextEncode.define_schema().display_name

    assert zen_nodes.NODE_SUFFIX in adapter
    assert zen_nodes.NODE_SUFFIX in encode


def test_both_nodes_are_suffixed_consistently():
    adapter_id, encode_id = _ids()

    assert adapter_id.endswith(encode_id[len("ZenImage21TextEncode"):]), (
        "both nodes must use the same suffix — a mixed pair would be confusing")
