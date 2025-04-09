"""Grids for 2D/3D image manipulations in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-grid-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_grid_utils.fftfreq_grid import fftfreq_grid
from torch_grid_utils.coordinate_grid import coordinate_grid
from torch_grid_utils.fftfreq_grid import dft_center, rfft_shape
from torch_grid_utils.shapes_2d import circle, rectangle, square, wedge
from torch_grid_utils.shapes_3d import sphere, cuboid, cube, cone

__all__ = [
    "fftfreq_grid",
    "coordinate_grid",
    "dft_center",
    "rfft_shape",
    "circle",
    "rectangle",
    "square",
    "wedge",
    "sphere",
    "cuboid",
    "cube",
    "cone",
]
