from typing import Sequence

import einops
import numpy as np
import torch


def coordinate_grid(
    image_shape: Sequence[int],
    origin: torch.Tensor | tuple[float, ...] | None = None,
    norm: bool = False,
    device: torch.device | None = None,
) -> torch.FloatTensor:
    """Get a dense grid of array coordinates from grid dimensions.

    e.g. for input `image_shape` of `(d, h, w)`, this function produces a
    `(d, h, w, 3)` grid of 3D coordinates.

    Coordinate order matches the order of dimensions in `image_shape`.

    i.e. `grid[..., 0]` contains coordinates in the `d` dimension.

    Parameters
    ----------
    image_shape: Sequence[int]
        Shape of the image for which coordinates should be returned.
    origin: torch.Tensor | tuple[float, ...] | None
        Origin of coordinate system. If `None`, default to placing the origin at
        index zero in all dimensions.
    norm: bool
        Whether to compute the Euclidean norm of the coordinate grid.
    device: torch.device
        PyTorch device on which to put the coordinate grid.

    Returns
    -------
    grid: torch.LongTensor
        `(*image_shape, image_ndim)` array of coordinates if `norm` is `False`
        else `(*image_shape, )`.
    """
    grid = torch.tensor(
        np.indices(image_shape),
        device=device,
        dtype=torch.float32
    )  # (coordinates, *image_shape)
    grid = einops.rearrange(grid, 'coords ... -> ... coords')
    ndim = len(image_shape)
    if origin is not None:
        origin = torch.as_tensor(origin, dtype=grid.dtype, device=grid.device)
        origin = torch.atleast_1d(origin)
        origin, ps = einops.pack([origin], pattern='* coords')
        ones = ' '.join('1' * ndim)
        axis_ids = ' '.join(_unique_characters(ndim))
        origin = einops.rearrange(origin, f"b coords -> b {ones} coords")
        grid = grid - origin
        [grid] = einops.unpack(grid, packed_shapes=ps, pattern=f'* {axis_ids} coords')
    if norm is True:
        grid = einops.reduce(grid ** 2, '... coords -> ...', reduction='sum') ** 0.5
    return grid


def _unique_characters(n: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return chars[:n]
