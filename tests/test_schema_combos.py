"""The loader node must offer file/folder PICKERS, not free-text path boxes.

Rationale (Andrii, 2026-09-23): typing absolute paths into a widget is a relic.
ComfyUI's own loaders expose a dropdown built from the registered model folders
(`folder_paths.get_filename_list`), with an upload button for files that are not
there yet. Our loader should do the same for `adapter_file` and `text_encoder`,
while still accepting a plain string when the value arrives from an old workflow
or the API.

These tests pin the *contract* of the schema, not the internals: which inputs are
Combo (dropdown) and which stay String.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _loader_schema():
    import zen_nodes
    return zen_nodes.ZenImage21AdapterLoader.define_schema()


def _inputs(schema):
    return {p.kwargs.get("id") or (p.args[0] if p.args else None): p
            for p in schema.kwargs.get("inputs", [])}


def test_adapter_file_is_a_combo_not_a_text_box():
    """`adapter_file` must be a dropdown, so the user picks the file instead of typing."""
    inputs = _inputs(_loader_schema())
    spec = inputs["adapter_file"]
    # a Combo port carries `options`; a String port does not
    assert "options" in spec.kwargs, (
        "adapter_file is still a free-text field — it must be an io.Combo.Input "
        "with options built from the ComfyUI model folders"
    )
    assert isinstance(spec.kwargs["options"], (list, tuple)), "options must be a sequence"
    assert len(spec.kwargs["options"]) >= 1, "a Combo needs at least one option"


def test_text_encoder_is_a_combo_not_a_text_box():
    """`text_encoder` must be a dropdown of the folders under models/text_encoders/."""
    inputs = _inputs(_loader_schema())
    spec = inputs["text_encoder"]
    assert "options" in spec.kwargs, (
        "text_encoder is still a free-text field — it must offer the encoder folders "
        "as a dropdown (plus a remote id fallback)"
    )
    assert isinstance(spec.kwargs["options"], (list, tuple))
    assert len(spec.kwargs["options"]) >= 1


def test_combos_allow_upload_so_a_missing_file_can_be_added_from_the_ui():
    """Both pickers should accept a file upload — otherwise a fresh install is stuck."""
    inputs = _inputs(_loader_schema())
    adapter = inputs["adapter_file"]
    assert adapter.kwargs.get("upload") is not None, (
        "adapter_file has no upload handle: a user with no adapter yet cannot add it "
        "from the node UI (io.UploadType.model)"
    )


def test_model_folder_stays_optional_string():
    """`model_folder` points at an arbitrary checkout dir — not a models/ entry, so String."""
    inputs = _inputs(_loader_schema())
    spec = inputs["model_folder"]
    assert "options" not in spec.kwargs, "model_folder must stay a String (arbitrary checkout path)"
    assert spec.kwargs.get("optional") is True, "model_folder is an optional fallback"
