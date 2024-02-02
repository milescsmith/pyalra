"""
.. module:: pyalra
   :platform: Unix, Windows
   :synopsis: {{description}}

.. moduleauthor:: Miles Smith <miles-smith@omrf.org>
"""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from pyalra.alra import alra
from pyalra.choose_k import choose_k

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

logger.disable("pyalra")

__all__ = ["alra", "choose_k"]
