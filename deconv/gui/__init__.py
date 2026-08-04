"""Qt GUI package with lazy public imports.

Lazy imports avoid a circular dependency while ``legacy_runtime`` imports the
translated widget wrappers from this package.
"""
from __future__ import annotations

__all__ = [
    "DeconvolutionMainWindow",
    "LoadGenerateTab",
    "DegradedInputTab",
    "AlgorithmTab",
    "TestTab",
    "ImageCanvas",
]


def __getattr__(name: str):
    if name == "DeconvolutionMainWindow":
        from .main_window import DeconvolutionMainWindow
        return DeconvolutionMainWindow
    if name in {"LoadGenerateTab", "DegradedInputTab", "AlgorithmTab", "TestTab", "ImageCanvas"}:
        from .tabs import AlgorithmTab, DegradedInputTab, ImageCanvas, LoadGenerateTab, TestTab
        return {
            "LoadGenerateTab": LoadGenerateTab,
            "DegradedInputTab": DegradedInputTab,
            "AlgorithmTab": AlgorithmTab,
            "TestTab": TestTab,
            "ImageCanvas": ImageCanvas,
        }[name]
    raise AttributeError(name)
