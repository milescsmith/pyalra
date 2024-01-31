"""
.. module:: pyalra
   :platform: Unix, Windows
   :synopsis: {{description}}

.. moduleauthor:: Miles Smith <miles-smith@omrf.org>
"""

from importlib.metadata import PackageNotFoundError, version

from pyalra.alra import alra
from pyalra.choose_k import choose_k
from pyalra.logger import init_logger

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

init_logger(1)

__all__ = [
    "alra",
    "choose_k"
]
