"""Regression guards for existing pure logic.

These pass today. They exist so the validation work (which touches the same
module) cannot silently break the adapter-metadata parsing or the cache key.

No GPU, no model download: `_fusion_config` is pure string handling.
"""
import zen_nodes
from zen_nodes import _fusion_config


def test_fusion_config_parses_adapter_metadata():
    """adapter_v11.safetensors stores its config in safetensors metadata as strings."""
    meta = {
        "in_dim": "4096",
        "out_dim": "4096",
        "hidden": "4096",
        "proj_layers": "2",
        "norm": "none",
        "n_slices": "1",
        "student_layers": "[7,15,23]",
    }

    cfg = _fusion_config(meta)

    assert cfg["in_dim"] == 4096
    assert cfg["out_dim"] == 4096
    assert cfg["hidden"] == 4096
    assert cfg["proj_layers"] == 2
    assert cfg["n_slices"] == 1
    assert cfg["norm"] == "none"
    assert cfg["student_layers"] == [7, 15, 23]


def test_fusion_config_accepts_comma_separated_layers():
    """student_layers is seen both as '[7,15,23]' and as '7,15,23'."""
    meta = {
        "in_dim": "4096", "out_dim": "4096", "hidden": "4096",
        "proj_layers": "2", "norm": "none", "n_slices": "1",
        "student_layers": "7, 15, 23",
    }

    cfg = _fusion_config(meta)

    assert cfg["student_layers"] == [7, 15, 23]


def test_fusion_config_renames_attn_max_len_to_max_len():
    meta = {
        "in_dim": "4096", "out_dim": "4096", "hidden": "4096",
        "proj_layers": "2", "norm": "none", "n_slices": "1",
        "student_layers": "[7]", "attn_max_len": "256",
    }

    cfg = _fusion_config(meta)

    assert cfg["max_len"] == 256
    assert "attn_max_len" not in cfg


def test_fusion_config_applies_defaults():
    meta = {
        "in_dim": "4096", "out_dim": "4096", "hidden": "4096",
        "proj_layers": "2", "norm": "none", "n_slices": "1",
        "student_layers": "[7]",
    }

    cfg = _fusion_config(meta)

    assert cfg["max_len"] == 512
    assert cfg["mixer_ffn"] == 2
    assert cfg["drop_idx"] == 14


def test_fusion_config_rejects_non_adapter_metadata():
    """A wrong file must say so instead of failing later with a KeyError."""
    with __import__("pytest").raises(ValueError) as exc:
        _fusion_config({"foo": "bar"})

    msg = str(exc.value)
    assert "adapter" in msg.lower()


# --------------------------------------------------------------------------
# Cache: the loader keeps a module-level cache keyed by configuration. Every
# entry pins ~1.7 GB (encoder) + 0.6 GB (adapter) of VRAM, so it must be
# bounded — otherwise two or three different configurations exhaust a 12 GB
# card.
# --------------------------------------------------------------------------

def test_cache_is_bounded():
    assert hasattr(zen_nodes, "_CACHE_MAX"), (
        "the adapter cache must declare a maximum size")
    assert isinstance(zen_nodes._CACHE_MAX, int)
    assert 1 <= zen_nodes._CACHE_MAX <= 4, (
        "each entry costs ~2.3 GB of VRAM; more than a couple will not fit")
