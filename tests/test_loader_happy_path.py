"""The happy path must survive the validation work.

Validation that only rejects is useless if it also rejects valid input. These
tests build a real (tiny) adapter safetensors file and drive the actual load
path, so the message-improving changes cannot silently break a correct setup.
"""
import torch
from safetensors.torch import save_file

from zen_nodes import _load_fusion, resolve_adapter_path

ADAPTER_META = {
    "in_dim": "4",
    "out_dim": "4",
    "hidden": "8",
    "proj_layers": "1",
    "norm": "none",
    "n_slices": "1",
    "student_layers": "[1]",
}


def _make_adapter(path):
    save_file({"proj.weight": torch.zeros(2, 2)}, str(path), metadata=dict(ADAPTER_META))
    return path


def test_valid_adapter_file_is_read(tmp_path):
    path = _make_adapter(tmp_path / "adapter_v11.safetensors")

    config, state = _load_fusion(str(path))

    assert config["in_dim"] == 4
    assert config["student_layers"] == [1]
    assert "proj.weight" in state


def test_valid_adapter_survives_a_relative_path(tmp_path, monkeypatch):
    """A relative path that exists must resolve — not be rejected as a `models/` miss."""
    _make_adapter(tmp_path / "adapter_v11.safetensors")
    monkeypatch.chdir(tmp_path)

    config, state = _load_fusion("adapter_v11.safetensors")

    assert config["in_dim"] == 4
    assert "proj.weight" in state


def test_resolve_adapter_path_accepts_absolute_and_relative(tmp_path, monkeypatch):
    path = _make_adapter(tmp_path / "adapter_v11.safetensors")

    assert resolve_adapter_path(str(path)) == str(path)

    monkeypatch.chdir(tmp_path)
    assert resolve_adapter_path("adapter_v11.safetensors") == str(path)


def test_resolve_adapter_path_returns_none_for_missing(tmp_path):
    assert resolve_adapter_path(str(tmp_path / "nope.safetensors")) is None
    assert resolve_adapter_path("") is None
