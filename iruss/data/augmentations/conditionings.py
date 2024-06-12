from typing import List, Tuple, Union

import torch
from positional_encodings.torch_encodings import PositionalEncoding2D
from torch import Tensor
from typeguard import typechecked

from ..augmentations.helpers import _parse_size

__all__ = ["sum_index_matrix", "positional_encoding"]


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
    h, w = _parse_size(size)
    p_enc_2d = PositionalEncoding2D(num_channels)

    y = torch.zeros((1, h, w, num_channels))
    enc = p_enc_2d(y).squeeze(0)

    return enc
