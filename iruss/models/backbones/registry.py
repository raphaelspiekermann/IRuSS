from functools import partial
from typing import Callable, Optional

from torch import nn

from .backbone import Backbone

_BUILDERS = {}


def register_model(model: Callable[..., nn.Module], /, *, name: Optional[str] = None):
    name = name or model.__class__.__name__
    assert name not in _BUILDERS
    _BUILDERS[name] = model


def get_model(model: str) -> Backbone:
    model = _BUILDERS[model]()
    assert isinstance(model, Backbone)

    return model


def list_models() -> list[str]:
    return sorted(_BUILDERS.keys())
