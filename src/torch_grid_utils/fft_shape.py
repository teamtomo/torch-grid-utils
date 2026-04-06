"""FFT-related shape utilities (rFFT output shape, FFT-friendly padding sizes)."""

import math
from typing import Sequence

import torch


def rfft_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an ``rfft`` on an input with ``input_shape``."""
    out = list(input_shape)
    out[-1] = int((out[-1] / 2) + 1)
    return tuple(out)


def next_fft_size(n: int) -> int:
    """Smallest integer ``>= n`` whose prime factors are only 2, 3, 5, and 7.

    Such sizes are convenient for FFT implementations that pad to "FFT-friendly"
    lengths (mixed-radix transforms). For ``n <= 1``, returns ``1``.

    Parameters
    ----------
    n
        Minimum required size (typically a padded image edge length).

    Returns
    -------
    int
        The minimal integer at least ``n`` whose prime factors lie in ``{2, 3, 5, 7}``.
    """
    if n <= 1:
        return 1

    limit = max(2 * n, 32)
    while True:
        i_max = int(math.floor(math.log2(limit))) + 1
        j_max = int(math.floor(math.log(limit) / math.log(3))) + 1
        k_max = int(math.floor(math.log(limit) / math.log(5))) + 1
        l_max = int(math.floor(math.log(limit) / math.log(7))) + 1

        i = torch.arange(i_max, dtype=torch.int64)
        j = torch.arange(j_max, dtype=torch.int64)
        k = torch.arange(k_max, dtype=torch.int64)
        e7 = torch.arange(l_max, dtype=torch.int64)
        ii, jj, kk, ll = torch.meshgrid(i, j, k, e7, indexing="ij")
        vals = (2**ii) * (3**jj) * (5**kk) * (7**ll)
        candidates = vals[vals >= n]
        if candidates.numel() > 0:
            return int(candidates.min().item())
        limit *= 2
