import numbers
from collections.abc import Sequence
from typing import List, Tuple, Union

import torch
import torchvision.transforms.functional as F
from PIL import Image
from src.utils.type_checking import typechecked
from torch import Tensor
from torchvision import transforms


def _setup_size(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be a sequence of length {msg}.")

    return [float(d) for d in x]


@typechecked
def _parse_size(size: Union[int, Tuple, List]) -> Tuple[int, int]:
    if isinstance(size, int):
        size = (size, size)
    if isinstance(size, list):
        size = tuple(size)
    if isinstance(size, tuple):
        assert len(size) == 2
        for s in size:
            assert isinstance(s, int) and s > 0
    else:
        raise ValueError(f"Invalid size {size}")
    return size


def _download_samples_images():
    import requests

    url1 = "https://tu-dortmund.sciebo.de/s/ZcHkwmyFTzou5iq/download"
    url2 = "https://tu-dortmund.sciebo.de/s/MEocJGnlz53Z3yv/download"
    url3 = "https://tu-dortmund.sciebo.de/s/ejxrfTj3vf9gxiR/download"

    image1 = Image.open(requests.get(url1, stream=True, timeout=5).raw)
    image2 = Image.open(requests.get(url2, stream=True, timeout=5).raw)
    image3 = Image.open(requests.get(url3, stream=True, timeout=5).raw)

    return [image1, image2, image3]


INTERPOLATION_MODES = {
    "nearest": F.InterpolationMode.NEAREST,
    "bilinear": F.InterpolationMode.BILINEAR,
    "bicubic": F.InterpolationMode.BICUBIC,
    "lanczos": F.InterpolationMode.LANCZOS,
}


def _get_interpolation_mode(
    mode: Union[str, transforms.InterpolationMode]
) -> transforms.InterpolationMode:
    if isinstance(mode, str):
        return INTERPOLATION_MODES[mode]
    elif isinstance(mode, transforms.InterpolationMode):
        return mode
    else:
        raise ValueError(f"Invalid interpolation mode: {mode}")


@typechecked
def sum_index_matrix(size: Union[int, Tuple, List]) -> Tensor:
    """Creates a tensor where each element is the sum of its indices.

    Args:
        size (int | Tuple | List): Size of the tensor. Tuples are interpreted as (height, width).

    Returns:
        Tensor: Matrix with the sum of its indices.

    Example:
    >>> sum_index_matrix(3)
    tensor([[0, 1, 2],
            [1, 2, 3],
            [2, 3, 4]])

    >>> sum_index_matrix((3, 4))
    tensor([[0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5]])
    """

    h, w = _parse_size(size)
    return torch.arange(h).unsqueeze(1) + torch.arange(w).unsqueeze(0)


@typechecked
def positional_encoding(size: Union[int, Tuple, List], num_channels) -> Tensor:
    from positional_encodings.torch_encodings import PositionalEncoding2D

    n, m = _parse_size(size)
    p_enc_2d = PositionalEncoding2D(num_channels)

    y = torch.zeros((1, n, m, num_channels))
    enc = p_enc_2d(y).squeeze(0)

    return enc
