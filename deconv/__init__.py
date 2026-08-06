"""GUI and reusable Python API for grayscale image deconvolution."""

__version__ = "0.106.0"

from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all

__all__ = ["__version__", *_api_all]
