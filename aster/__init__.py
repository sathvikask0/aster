"""Aster: experimental biological decision models.

`AsterConfig` / `AsterModel` are imported lazily so that the torch-free parts of
the package (the benchmark loader and the shortcut-ceiling estimator) can be
used for analysis without pulling in torch and transformers.
"""

__all__ = ["AsterConfig", "AsterModel"]


def __getattr__(name):
    if name in __all__:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
