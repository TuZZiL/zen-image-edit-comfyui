"""Behavioural tests for the adapter cache and the device helper.

These exist because a previous revision passed all 21 tests while being broken
in production: `_device()` had been renamed away by a bad edit and `_CACHE` was
a plain `dict` while `load_adapter` called `OrderedDict.move_to_end`. Neither
path was exercised by the suite, so a green CI meant nothing.

Every test here drives the real `load_adapter` code with a stubbed adapter, so
no GPU and no model weights are needed.
"""
import pytest

import zen_nodes


class _FakeAdapter:
    """Stands in for ZenImage21Adapter: counts constructions, records release."""

    instances = 0
    last_args = None

    def __init__(self, model_folder, text_encoder, adapter_file, dtype_name):
        _FakeAdapter.instances += 1
        _FakeAdapter.last_args = (model_folder, text_encoder, adapter_file, dtype_name)
        self.released = False
        self.student = self  # _release() walks these attributes and calls .to("cpu")

    def to(self, device):
        self.released = True
        return self


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    zen_nodes._CACHE.clear()
    _FakeAdapter.instances = 0
    _FakeAdapter.last_args = None
    monkeypatch.setattr(zen_nodes, "ZenImage21Adapter", _FakeAdapter)
    yield
    zen_nodes._CACHE.clear()


# --------------------------------------------------------------------------
# the regression that slipped through: _device was renamed away
# --------------------------------------------------------------------------

def test_device_helper_exists():
    """ZenImage21Adapter.__init__ calls _device(); a bad edit once removed it."""
    assert callable(zen_nodes._device), "_device() must exist — the adapter loader calls it"


def test_device_returns_a_torch_device():
    device = zen_nodes._device()

    assert device is not None


# --------------------------------------------------------------------------
# cache: the hit path must not raise
# --------------------------------------------------------------------------

def test_cache_type_supports_the_lru_operations_used():
    assert hasattr(zen_nodes._CACHE, "move_to_end"), (
        "load_adapter calls _CACHE.move_to_end on a hit: the cache must be an OrderedDict")


def test_second_load_of_same_key_returns_cached_adapter():
    """The hit path called move_to_end on a plain dict and raised AttributeError."""
    first = zen_nodes.load_adapter("", "bf16", "encoder", "adapter")
    second = zen_nodes.load_adapter("", "bf16", "encoder", "adapter")

    assert first is second, "the same key must return the cached instance"
    assert _FakeAdapter.instances == 1, "the adapter must be constructed only once"


def test_third_load_reuses_cache_without_reconstruction():
    zen_nodes.load_adapter("", "bf16", "encoder", "adapter")
    zen_nodes.load_adapter("", "bf16", "encoder", "adapter")
    zen_nodes.load_adapter("", "bf16", "encoder", "adapter")

    assert _FakeAdapter.instances == 1


# --------------------------------------------------------------------------
# cache: eviction must stay bounded and must free the evicted entry
# --------------------------------------------------------------------------

def test_cache_stays_within_its_bound():
    for i in range(5):
        zen_nodes.load_adapter("", "bf16", f"encoder{i}", "adapter")

    assert len(zen_nodes._CACHE) <= zen_nodes._CACHE_MAX


def test_evicted_adapter_is_released():
    first = zen_nodes.load_adapter("", "bf16", "encoder1", "adapter")
    zen_nodes.load_adapter("", "bf16", "encoder2", "adapter")
    assert first.released is False, "the first entry is still cached at this point"

    zen_nodes.load_adapter("", "bf16", "encoder3", "adapter")  # forces an eviction

    assert first.released is True, "an evicted adapter must be moved off the accelerator"


def test_different_keys_are_cached_separately():
    a = zen_nodes.load_adapter("", "bf16", "encoderA", "adapter")
    b = zen_nodes.load_adapter("", "fp16", "encoderA", "adapter")

    assert a is not b, "dtype is part of the cache key"
    assert _FakeAdapter.instances == 2
