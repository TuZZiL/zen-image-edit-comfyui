"""`ZEN_NODE_SUFFIX` is one env var, but it must not force one id on two checkouts.

The suffix exists so this fork can be installed next to the upstream node: two
custom nodes claiming the same ``NODE_CLASS_MAPPINGS`` key collide, and ComfyUI's
loader silently keeps the last one (no sort in ``os.listdir``, so "last" depends
on the filesystem). Suffixing the ids fixes that — for *one* fork checkout.

It does not fix two checkouts of *this* fork. ``NODE_SUFFIX`` is read once at
module import, and every checkout imports inside the same ComfyUI process, so
both read the same ``os.environ`` and both take the same suffix. ``__init__.py``
already loads the sibling modules under per-repo names, but the node ids are not
isolated that way: the README promise ("set it to any string to label a second
or third checkout") does not hold for a second checkout of the fork itself.

So the suffix needs a per-checkout source. This file locks in the resolution
order: explicit env > a small file next to this repo's ``zen_nodes.py`` >
the historical ``"Plus"`` default.
"""
import importlib

import pytest

import zen_nodes


@pytest.fixture
def fresh_module():
    """`NODE_SUFFIX` is computed at import; reload to exercise the resolver.

    Reload is not guaranteed to work here: another test module pops
    ``sys.modules["zen_nodes"]`` while cleaning up after itself (it plants a
    decoy under that name), so this fixture must survive both a present and an
    absent module instead of assuming either.
    """
    import sys

    def _fresh():
        module = sys.modules.get("zen_nodes")
        if module is None:
            return importlib.import_module("zen_nodes")
        return importlib.reload(module)

    yield _fresh

    module = sys.modules.get("zen_nodes")
    if module is not None:
        importlib.reload(module)


def test_resolver_exists_and_reads_the_checkout_file(tmp_path):
    """A checkout-local file names this checkout, without touching the env."""
    (tmp_path / ".zen_node_suffix").write_text("A\n", encoding="utf-8")

    assert zen_nodes.resolve_node_suffix({}, tmp_path) == "A"


def test_resolver_strips_surrounding_whitespace(tmp_path):
    """Trailing newline / spaces from an editor must not leak into a node id."""
    (tmp_path / ".zen_node_suffix").write_text("  Beta  \n", encoding="utf-8")

    assert zen_nodes.resolve_node_suffix({}, tmp_path) == "Beta"


def test_explicit_env_beats_the_checkout_file(tmp_path):
    """The env var stays a global override, so existing installs keep working."""
    (tmp_path / ".zen_node_suffix").write_text("A\n", encoding="utf-8")

    assert zen_nodes.resolve_node_suffix({"ZEN_NODE_SUFFIX": "Forced"}, tmp_path) == "Forced"


def test_empty_env_still_strips_the_suffix(tmp_path):
    """`ZEN_NODE_SUFFIX=""` means upstream ids; a file must not resurrect a suffix."""
    (tmp_path / ".zen_node_suffix").write_text("A\n", encoding="utf-8")

    assert zen_nodes.resolve_node_suffix({"ZEN_NODE_SUFFIX": ""}, tmp_path) == ""


def test_default_is_plus_with_neither_env_nor_file(tmp_path):
    """Unchanged behaviour for every single-checkout install in the wild."""
    assert zen_nodes.resolve_node_suffix({}, tmp_path) == "Plus"


def test_blank_checkout_file_falls_back_to_the_default(tmp_path):
    """An empty file is an editor accident, not a request for upstream ids."""
    (tmp_path / ".zen_node_suffix").write_text("\n", encoding="utf-8")

    assert zen_nodes.resolve_node_suffix({}, tmp_path) == "Plus"


def test_two_checkouts_get_two_different_suffixes(tmp_path):
    """The whole point: the same env, two repos, ids that no longer collide."""
    first = tmp_path / "zen-image-edit"
    second = tmp_path / "zen-image-edit-plus"
    first.mkdir()
    second.mkdir()
    (first / ".zen_node_suffix").write_text("First\n", encoding="utf-8")
    (second / ".zen_node_suffix").write_text("Second\n", encoding="utf-8")

    env = {}  # identical environment, as inside one ComfyUI process
    id_a = f"ZenImage21AdapterLoader{zen_nodes.resolve_node_suffix(env, first)}"
    id_b = f"ZenImage21AdapterLoader{zen_nodes.resolve_node_suffix(env, second)}"

    assert id_a != id_b, (
        "two checkouts of this fork still claim the same NODE_CLASS_MAPPINGS key — "
        "the second one silently replaces the first")


def test_module_suffix_matches_the_resolver(fresh_module):
    """The shipped constant must be what the resolver returns for this repo."""
    module = fresh_module()

    expected = module.resolve_node_suffix(
        __import__("os").environ, __import__("pathlib").Path(module.__file__).parent)

    assert module.NODE_SUFFIX == expected


def test_registration_log_names_the_checkout_that_will_serve_the_ids():
    """A collision used to be invisible. "Which implementation runs" must be
    answerable by grepping the startup log, not by a filesystem experiment."""
    line = zen_nodes.registration_log_line(
        ["ZenImage21AdapterLoaderPlus", "ZenImage21TextEncodePlus"],
        "/opt/ComfyUI/custom_nodes/zen-image-edit")

    assert "zen-image-edit" in line, "the log must name the checkout directory"
    assert "ZenImage21AdapterLoaderPlus" in line
    assert "ZenImage21TextEncodePlus" in line


def test_get_node_list_emits_the_registration_log(capsys):
    """The log has to fire where ComfyUI actually loads the node, or it is dead code."""
    import asyncio

    extension = asyncio.run(zen_nodes.comfy_entrypoint())
    node_ids = [cls.define_schema().node_id
                for cls in asyncio.run(extension.get_node_list())]

    out = capsys.readouterr().out
    for node_id in node_ids:
        assert node_id in out, f"{node_id} was not named in the startup log"
