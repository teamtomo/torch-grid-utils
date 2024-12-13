# torch-grid-utils

[![License](https://img.shields.io/pypi/l/torch-grid-utils.svg?color=green)](https://github.com/alisterburt/torch-grid-utils/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-grid-utils.svg?color=green)](https://pypi.org/project/torch-grid-utils)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-grid-utils.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/torch-grid-utils/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/torch-grid-utils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/torch-grid-tuils/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/torch-grid-utils)

*torch-grid-utils* provides grid utilities for 2D/3D image manipulations in PyTorch.

## Installation

*torch-grid-utils* is available on the Python package index 
([PyPI](https://pypi.org/project/torch-grid-utils/)).

```shell
pip install torch-grid-utils
```

## Usage

*torch-grid-utils* provides two functions


[`torch_grid_utils.coordinate_grid`](./usage/coordinate_grid)

:   `coordinate_grid` generates a tensor containing the coordinates for each 
    pixel/voxel in a 2D/3D image.

[`torch_grid_utils.fftfreq_grid`](./usage/fftfreq_grid)

:   `fftfreq_grid` generates a tensor containing the sample frequencies for each 
    element of the discrete Fourier transform of a 2D/3D image.


