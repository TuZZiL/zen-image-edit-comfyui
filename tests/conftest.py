"""Make ``zen_nodes`` importable without a ComfyUI install.

``zen_nodes`` imports ``comfy.model_management``, ``comfy.utils`` and
``comfy_api.latest`` at module level, so plain ``import zen_nodes`` fails
anywhere ComfyUI is not installed — which is every CI runner, and any machine
that only wants to run the pure-function tests.

Design constraint learned the hard way: **never install a permissive
``__getattr__`` on a stub module.** torch calls ``inspect`` during its own
import, and ``inspect`` asks modules for ``__file__``; a stub that answers with
a catch-all object breaks `import torch` with a baffling
``TypeError: expected str, bytes or os.PathLike object``. So the stubs below
declare exactly the names the node classes touch, and nothing else.

Only names evaluated at *import time* matter here — the node bodies
(``define_schema``, ``execute``) run later, inside ComfyUI, and are not
exercised by these tests.
"""
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Port:
    """Stand-in for an io node port (``io.Image.Input(...)`` and friends)."""

    def __init__(self, *args, **kwargs):
        pass


class _Spec:
    """Stand-in for a schema/port factory (``io.String``, ``io.Custom``, ...)."""

    # io.Autogrow.Type is used as a runtime type annotation in the node signature.
    # Python 3.14 defers annotation evaluation (PEP 649) so it is never read there,
    # but 3.12 (the CI interpreter) evaluates it at class-creation time. Define it
    # so the stub does not pass locally and fail on CI.
    Type = object

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Spec()

    @staticmethod
    def Input(*args, **kwargs):
        return _Port()

    @staticmethod
    def Output(*args, **kwargs):
        return _Port()

    @staticmethod
    def TemplateNames(*args, **kwargs):
        return _Port()


class _Schema:
    """Remembers the schema kwargs so tests can assert on node_id/display_name."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.node_id = kwargs.get("node_id")
        self.display_name = kwargs.get("display_name")


class _ComfyNode:
    """Real class: nodes subclass it (``class Foo(io.ComfyNode)``)."""


class _ComfyExtension:
    """Real class: the extension subclasses it."""


class _IO(types.ModuleType):
    """``comfy_api.latest.io`` — only the names used at class-definition time."""

    __file__ = __file__  # keep inspect happy; torch reads this during import

    ComfyNode = _ComfyNode
    NodeOutput = _Spec
    Schema = _Schema
    String = _Spec
    Combo = _Spec
    Custom = _Spec
    Vae = _Spec
    Int = _Spec
    Image = _Spec
    Conditioning = _Spec
    Latent = _Spec
    Autogrow = _Spec


class _StubModule(types.ModuleType):
    __file__ = __file__  # see _IO.__file__ above


def _install_stubs():
    if "comfy" in sys.modules:
        return  # a real ComfyUI is present; leave it alone

    comfy = _StubModule("comfy")
    comfy_mm = _StubModule("comfy.model_management")
    comfy_utils = _StubModule("comfy.utils")

    def get_torch_device():
        import torch
        return torch.device("cpu")

    def intermediate_device():
        import torch
        return torch.device("cpu")

    setattr(comfy_mm, "get_torch_device", get_torch_device)
    setattr(comfy_mm, "intermediate_device", intermediate_device)

    latest = _StubModule("comfy_api.latest")
    latest_io = _IO("comfy_api.latest.io")
    setattr(latest, "io", latest_io)
    setattr(latest, "ComfyExtension", _ComfyExtension)

    comfy_api = _StubModule("comfy_api")
    setattr(comfy_api, "latest", latest)

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = comfy_mm
    sys.modules["comfy.utils"] = comfy_utils
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest


_install_stubs()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
