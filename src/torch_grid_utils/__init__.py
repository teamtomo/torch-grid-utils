"""Grids for 2D/3D image manipulations in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-grid-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_grid_utils.coordinate_grid import coordinate_grid, image_center
from torch_grid_utils.fftfreq_grid import (
    fftfreq_grid,
    dft_center,
    rfft_shape,
    fftshift_1d,
    ifftshift_1d,
    fftshift_2d,
    ifftshift_2d,
    fftshift_3d,
    ifftshift_3d,
    fftfreq_to_spatial_frequency,
    spatial_frequency_to_fftfreq,
)
from torch_grid_utils.shapes_2d import circle, rectangle, square, wedge
from torch_grid_utils.shapes_3d import sphere, cuboid, cube, cone

__all__ = [
    "coordinate_grid",
    "fftfreq_grid",
    "image_center",
    "dft_center",
    "rfft_shape",
    "fftshift_1d",
    "ifftshift_1d",
    "fftshift_2d",
    "ifftshift_2d",
    "fftshift_3d",
    "ifftshift_3d",
    "fftfreq_to_spatial_frequency",
    "spatial_frequency_to_fftfreq",
    "circle",
    "rectangle",
    "square",
    "wedge",
    "sphere",
    "cuboid",
    "cube",
    "cone",
]
