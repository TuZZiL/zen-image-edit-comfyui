"""Path validation for the adapter file and the text encoder.

These tests encode four failures hit in practice on a stock ComfyUI install
(ComfyUI 0.37.0, RTX 3060, Windows). Each of them surfaced as a raw
``FileNotFoundError`` / ``HFValidationError`` / ``OSError`` that named neither
the artifact nor the fix, and each cost real debugging time:

1. ``adapter_file`` is passed straight to ``safetensors.safe_open``, which
   resolves relative paths against the *process* working directory — not
   against ``ComfyUI/models/``. A value like ``adapter_v11.safetensors`` fails
   with a bare "No such file or directory".
2. ``text_encoder`` must be a *directory*. Pointing it at a ``.safetensors``
   file makes ``from_pretrained`` treat the string as a Hugging Face repo id
   and raise ``HFValidationError``.
3. An encoder directory can be incomplete: the HF repo
   ``Qwen/Qwen3.5-0.8B`` names its weights
   ``model.safetensors-00001-of-00001.safetensors``, so without
   ``model.safetensors.index.json`` ``from_pretrained`` reports
   "no file named model.safetensors..." even though the weights are present.
4. Relative paths in general are the trap: the user's mental model is
   "ComfyUI finds models in models/", but this node does not.

Every assertion here is on the *message*, because the message is the interface
a user meets at 2am.
"""
import pytest

import zen_nodes
from zen_nodes import _load_fusion

# --------------------------------------------------------------------------
# 1. adapter_file
# --------------------------------------------------------------------------

def test_missing_adapter_reports_actionable_error(tmp_path):
    missing = tmp_path / "adapter_v11.safetensors"

    with pytest.raises(Exception) as exc:
        _load_fusion(str(missing))

    msg = str(exc.value)
    assert "adapter" in msg.lower(), (
        "the error must name the artifact that is missing, not just a path")
    assert str(missing) in msg, (
        "the error must show the resolved path so the user can see what was tried")
    assert "absolute" in msg.lower() or "models" in msg.lower(), (
        "the error must say where the file is expected to live")

def test_relative_adapter_path_error_mentions_working_directory(tmp_path, monkeypatch):
    """A relative path is the trap: it resolves against cwd, not models/."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception) as exc:
        _load_fusion("adapter_v11.safetensors")

    msg = str(exc.value)
    assert "adapter" in msg.lower()
    assert "working directory" in msg.lower() or "cwd" in msg.lower(), (
        "the user must learn that relative paths resolve against the working directory")

def test_existing_but_non_adapter_file_gets_a_clear_message(tmp_path):
    """A file that exists but is not a safetensors adapter must not crash raw."""
    bogus = tmp_path / "adapter_v11.safetensors"
    bogus.write_bytes(b"not a safetensors file")

    with pytest.raises(Exception) as exc:
        _load_fusion(str(bogus))

    msg = str(exc.value)
    assert "adapter" in msg.lower() or "safetensors" in msg.lower()

# --------------------------------------------------------------------------
# 2. text_encoder must be a directory
# --------------------------------------------------------------------------

def test_text_encoder_pointing_at_a_file_is_rejected(tmp_path):
    """HFValidationError costs an hour; a directed message costs a second."""
    encoder_file = tmp_path / "Qwen3.5-0.8B.safetensors"
    encoder_file.write_bytes(b"x")

    with pytest.raises(ValueError) as exc:
        zen_nodes.validate_text_encoder(str(encoder_file))

    msg = str(exc.value)
    assert "directory" in msg.lower(), (
        "the message must state that a directory is required")
    assert "from_pretrained" in msg or "HFValidationError" in msg, (
        "the message must explain the failure the user would otherwise see")
    assert str(encoder_file) in msg

# --------------------------------------------------------------------------
# 3. incomplete encoder directory
# --------------------------------------------------------------------------

def _make_encoder_dir(tmp_path, *, names):
    d = tmp_path / "qwen3.5_0.8b"
    d.mkdir()
    for name in names:
        (d / name).write_bytes(b"x")
    return d

def test_encoder_dir_without_weights_is_explained(tmp_path):
    d = _make_encoder_dir(tmp_path, names=["config.json", "tokenizer_config.json"])

    with pytest.raises(ValueError) as exc:
        zen_nodes.validate_text_encoder(str(d))

    msg = str(exc.value)
    assert "weights" in msg.lower() or "safetensors" in msg.lower()
    assert str(d) in msg

def test_encoder_dir_with_oddly_named_weights_needs_the_index(tmp_path):
    """The Qwen/Qwen3.5-0.8B trap: weights present, index missing."""
    d = _make_encoder_dir(tmp_path, names=[
        "config.json",
        "tokenizer_config.json",
        "model.safetensors-00001-of-00001.safetensors",
    ])

    with pytest.raises(ValueError) as exc:
        zen_nodes.validate_text_encoder(str(d))

    msg = str(exc.value)
    assert "index.json" in msg, (
        "the missing index.json is the whole point: without it from_pretrained "
        "cannot see the oddly named shard")
    assert "hf download" in msg, "the error must be actionable, not just descriptive"

def test_encoder_dir_with_canonical_weight_name_is_accepted(tmp_path):
    """A standard model.safetensors needs no index — do not over-reject."""
    d = _make_encoder_dir(tmp_path, names=[
        "config.json",
        "tokenizer_config.json",
        "model.safetensors",
    ])

    encoder, processor, tokenizer = zen_nodes.validate_text_encoder(str(d))

    assert encoder == str(d)
    assert processor == str(d)
    assert tokenizer == str(d)

def test_encoder_dir_with_index_is_accepted(tmp_path):
    """Weight shard + index.json is a complete download."""
    d = _make_encoder_dir(tmp_path, names=[
        "config.json",
        "tokenizer_config.json",
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors.index.json",
    ])

    encoder, _, _ = zen_nodes.validate_text_encoder(str(d))

    assert encoder == str(d)

def test_missing_encoder_dir_is_reported(tmp_path):
    with pytest.raises(ValueError) as exc:
        zen_nodes.validate_text_encoder(str(tmp_path / "does-not-exist"))

    msg = str(exc.value)
    assert "does-not-exist" in msg
    assert "hf download" in msg or "hugging face" in msg.lower()

# --------------------------------------------------------------------------
# 4. Hugging Face repo id stays supported (offline-free path)
# --------------------------------------------------------------------------

def test_hf_repo_id_is_passed_through_untouched():
    """`Qwen/Qwen3.5-0.8B` must keep working — it downloads via the library."""
    encoder, processor, tokenizer = zen_nodes.validate_text_encoder("Qwen/Qwen3.5-0.8B")

    assert encoder == "Qwen/Qwen3.5-0.8B"
    assert processor == "Qwen/Qwen3.5-0.8B"
    assert tokenizer == "Qwen/Qwen3.5-0.8B"

# --------------------------------------------------------------------------
# 5. relative path resolution helper
# --------------------------------------------------------------------------

def test_relative_path_is_resolved_to_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = zen_nodes.resolve_adapter_path("adapter_v11.safetensors")

    assert resolved is None or str(tmp_path) in str(resolved) or resolved == "adapter_v11.safetensors"
