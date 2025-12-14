"""Functions to construct grids of DFT sample frequencies."""

from typing import Sequence, cast

import einops
import torch


# no lru_cache as it interferes with gradient calculation (see torch-fourier-shift PR#5)
def fftfreq_grid(
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    spacing: float | tuple[float, float] | tuple[float, float, float] = 1,
    norm: bool = False,
    device: torch.device | None = None,
    transform_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct a 2D or 3D grid of DFT sample frequencies.

    For a 2D image with shape `(h, w)` and `rfft=False` this function will produce
    a `(h, w, 2)` array of DFT sample frequencies in the `h` and `w` dimensions.
    If `norm` is True the Euclidean norm will be calculated over the last dimension
    leaving a `(h, w)` grid.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        Shape of the 2D or 3D image before computing the DFT.
    rfft: bool
        Whether the output should contain frequencies for a real-valued DFT.
    fftshift: bool
        Whether to fftshift the output grid.
    spacing: float | tuple[float, float] | tuple[float, float, float]
        Spacing between samples in each dimension. Sampling is considered to be
        isotropic if a single value is passed.
    norm: bool
        Whether to compute the Euclidean norm over the last dimension.
    device: torch.device | None
        PyTorch device on which the returned grid will be stored.
    transform_matrix: torch.Tensor | None
        Optional 2x2 transformation matrix for anisotropic magnification
        (2D images only). Applied to the frequency grid as:
        [freq_y_new, freq_x_new]^T = transform_matrix @ [freq_y, freq_x]^T.
        If provided, must be a 2x2 torch.Tensor. For 3D images, this parameter
        is ignored.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(*image_shape, ndim)` array of DFT sample frequencies in each
        image dimension if `norm` is `False` else `(*image_shape, )`.
    """
    if len(image_shape) == 2:
        # Type narrowing: for 2D, spacing can only be float or tuple[float, float]
        spacing_2d = cast("float | tuple[float, float]", spacing)
        frequency_grid = _construct_fftfreq_grid_2d(
            image_shape=image_shape,
            rfft=rfft,
            spacing=spacing_2d,
            device=device,
        )
        # Apply transformation matrix in fftshifted coordinate system
        # (centered coordinates where DC is at center)
        if transform_matrix is not None:
            # Always shift to centered coordinates for transformation
            frequency_grid = einops.rearrange(frequency_grid, "... freq -> freq ...")
            frequency_grid = fftshift_2d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, "freq ... -> ... freq")
            # Apply transformation in centered coordinate system
            frequency_grid = _apply_transform_matrix_2d(
                frequency_grid, transform_matrix, device=device
            )
            # Undo shift if fftshift was False (user doesn't want final output shifted)
            if fftshift is False:
                frequency_grid = einops.rearrange(
                    frequency_grid, "... freq -> freq ..."
                )
                frequency_grid = ifftshift_2d(frequency_grid, rfft=rfft)
                frequency_grid = einops.rearrange(
                    frequency_grid, "freq ... -> ... freq"
                )
        elif fftshift is True:
            # No transformation, just apply fftshift if requested
            frequency_grid = einops.rearrange(frequency_grid, "... freq -> freq ...")
            frequency_grid = fftshift_2d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, "freq ... -> ... freq")
    elif len(image_shape) == 3:
        # Type narrowing: for 3D, spacing can only be float or
        # tuple[float, float, float]
        spacing_3d = cast("float | tuple[float, float, float]", spacing)
        frequency_grid = _construct_fftfreq_grid_3d(
            image_shape=image_shape,
            rfft=rfft,
            spacing=spacing_3d,
            device=device,
        )
        if fftshift is True:
            frequency_grid = einops.rearrange(frequency_grid, "... freq -> freq ...")
            frequency_grid = fftshift_3d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, "freq ... -> ... freq")
    else:
        raise NotImplementedError(
            "Construction of fftfreq grids is currently only supported for "
            "2D and 3D images."
        )
    if norm is True:
        frequency_grid = (
            einops.reduce(
                frequency_grid**2, "... squared_freqs -> ...", reduction="sum"
            )
            ** 0.5
        )
    return frequency_grid


def _construct_fftfreq_grid_2d(
    image_shape: tuple[int, int],
    rfft: bool,
    spacing: float | tuple[float, float] = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Whether the frequency grid is for a real fft (rfft).
    spacing: float | tuple[float, float]
        Sample spacing in `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of
        the two dimensions of the grid.
    """
    dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 2
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    h, w = image_shape
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_yy = einops.repeat(freq_y, "h -> h w", w=w)
    freq_xx = einops.repeat(freq_x, "w -> h w", h=h)
    return einops.rearrange([freq_yy, freq_xx], "freq h w -> h w freq")


def _apply_transform_matrix_2d(
    frequency_grid: torch.Tensor,
    transform_matrix: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Apply a 2x2 transformation matrix to a 2D frequency grid.

    Parameters
    ----------
    frequency_grid: torch.Tensor
        Frequency grid with shape `(h, w, 2)`.
    transform_matrix: torch.Tensor
        2x2 transformation matrix.
    device: torch.device | None
        PyTorch device for the transformation matrix.

    Returns
    -------
    transformed_grid: torch.Tensor
        Transformed frequency grid with shape `(h, w, 2)`.
    """
    # Ensure matrix is on the correct device
    if device is not None:
        transform_matrix = transform_matrix.to(device)
    elif frequency_grid.device is not None:
        transform_matrix = transform_matrix.to(frequency_grid.device)

    # Validate matrix shape
    if transform_matrix.shape != (2, 2):
        raise ValueError(
            f"transform_matrix must be a 2x2 matrix, "
            f"got shape {transform_matrix.shape}"
        )

    # Ensure matrix is float type for matrix multiplication
    transform_matrix = transform_matrix.float()

    # Apply transformation:
    # [freq_y_new, freq_x_new]^T = transform_matrix @ [freq_y, freq_x]^T
    # frequency_grid has shape (h, w, 2), we need to apply matrix
    # multiplication to the last dimension. Reshape to (h*w, 2) for batch
    # matrix multiplication
    h, w = frequency_grid.shape[:2]
    freq_flat = einops.rearrange(frequency_grid, "h w freq -> (h w) freq")

    # Apply transformation: (h*w, 2) @ (2, 2)^T = (h*w, 2)
    # Note: transform_matrix @ freq_flat^T would give (2, h*w), so we
    # transpose
    transformed_flat = torch.matmul(freq_flat, transform_matrix.T)

    # Reshape back to (h, w, 2)
    transformed_grid = einops.rearrange(
        transformed_flat, "(h w) freq -> h w freq", h=h, w=w
    )

    return transformed_grid


def _construct_fftfreq_grid_3d(
    image_shape: Sequence[int],
    rfft: bool,
    spacing: float | tuple[float, float, float] = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 3D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 3D shape `(d, h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).
    spacing: float | tuple[float, float, float]
        Sample spacing in `d`, `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 3)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of dimensions
        of the grid.
    """
    dd, dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 3
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    d, h, w = image_shape
    freq_z = torch.fft.fftfreq(d, d=dd, device=device)
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    d, h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_zz = einops.repeat(freq_z, "d -> d h w", h=h, w=w)
    freq_yy = einops.repeat(freq_y, "h -> d h w", d=d, w=w)
    freq_xx = einops.repeat(freq_x, "w -> d h w", d=d, h=h)
    return einops.rearrange([freq_zz, freq_yy, freq_xx], "freq ... -> ... freq")


def rfft_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def dft_center(
    image_shape: tuple[int, ...],
    rfft: bool,
    fftshift: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape, device=device).float()
    if rfft is True:
        image_shape = torch.tensor(rfft_shape(image_shape), device=device)
    if fftshift is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode="floor")
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


def fftshift_1d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply fftshift to a 1D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, no shift is
        applied.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-1))
    else:
        output = input
    return output


def ifftshift_1d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply ifftshift to a 1D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, no shift is
        applied.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-1))
    else:
        output = input
    return output


def fftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply fftshift to a 2D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        dimension is shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-2,))
    return output


def ifftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply ifftshift to a 2D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        dimension is shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-2,))
    return output


def fftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply fftshift to a 3D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        two dimensions are shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-3, -2))
    return output


def ifftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply ifftshift to a 3D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        two dimensions are shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-3, -2))
    return output


def fftfreq_to_spatial_frequency(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per pixel to cycles per unit distance."""
    # cycles/px * px/distance = cycles/distance
    return torch.as_tensor(frequencies, dtype=torch.float32) * (1 / spacing)


def spatial_frequency_to_fftfreq(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per unit distance to cycles per pixel."""
    # cycles/distance * distance/px = cycles/px
    return torch.as_tensor(frequencies, dtype=torch.float32) * spacing
