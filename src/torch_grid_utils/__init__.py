"""Grids for 2D/3D image manipulations in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-grid-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .fftfreq_grid import fftfreq_grid
from .coordinate_grid import coordinate_grid
