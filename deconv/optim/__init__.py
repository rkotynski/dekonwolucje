"""Optimization helpers.

AutoTuneWorker is imported lazily to avoid a circular import with the Qt
compatibility module, which itself uses the isolated Auto numerical process.
"""
from .batch import BatchedScores
from .auto_process import AutoCancelledError, AutoProcessError, AutoNumericalProcessClient


def __getattr__(name):
    if name == "AutoTuneWorker":
        from .workers import AutoTuneWorker
        return AutoTuneWorker
    raise AttributeError(name)


__all__ = [
    "AutoTuneWorker",
    "BatchedScores",
    "AutoCancelledError",
    "AutoProcessError",
    "AutoNumericalProcessClient",
]
