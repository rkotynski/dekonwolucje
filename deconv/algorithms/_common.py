"""Shared numerical names used by algorithm implementations."""
from deconv.core import runtime as _runtime

for _name in _runtime.__all__:
    globals()[_name] = getattr(_runtime, _name)

__all__ = list(_runtime.__all__)
