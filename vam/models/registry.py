from __future__ import annotations

from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def _load_adapters() -> None:
    try:
        import vam.models.hf  # noqa: F401  (registers HF models; needs transformers)
    except ImportError:
        pass


def list_models() -> list[str]:
    _load_adapters()
    return sorted(_REGISTRY)


def get_model(name: str, **kwargs):
    _load_adapters()
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
