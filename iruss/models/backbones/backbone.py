import math
from abc import abstractmethod
from functools import lru_cache
from typing import Union

import torch
from torch import Tensor, nn

_SAMPLE_INPUT_DIM = 224


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, x: Tensor, return_feature_maps: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        raise NotImplementedError

    @lru_cache
    @torch.no_grad
    def _compute_sample_output(self):
        sample_input = torch.randn(2, self.in_channels, _SAMPLE_INPUT_DIM, _SAMPLE_INPUT_DIM)

        device = next(self.parameters()).device
        sample_input = sample_input.to(device)

        global_features, local_features = self.forward(sample_input, return_feature_maps=True)

        assert local_features.ndim == 4
        assert global_features.ndim == 2
        assert local_features.shape[1] == global_features.shape[1]

        return global_features, local_features

    def _probe_total_stride(self):
        _, local_features = self._compute_sample_output()

        if local_features.ndim == 3:
            # print(f"Vision Transformer output shape: {sample_output.shape}")
            # Vision Transformer -> reshape to B, C, H, W

            local_features = local_features[:, 1:]  # ignore CLS token
            local_features = local_features.permute(0, 2, 1)  # B, C, H*W

            b = local_features.shape[0]
            c = local_features.shape[1]
            d = int(math.sqrt(local_features.shape[-1]))

            assert d * d == local_features.shape[-1], "Expected square local features"

            local_features = local_features.reshape(b, c, d, d)

        assert local_features.ndim == 4, f"Expected 4D output, got {local_features}"
        # Convolutional Encoder -> dimensions are B, C, H, W
        h_out = local_features.shape[-2]
        w_out = local_features.shape[-1]

        h_in, w_in = _SAMPLE_INPUT_DIM, _SAMPLE_INPUT_DIM

        h_stride = h_in // h_out  # there could be a remainder
        w_stride = w_in // w_out

        return h_stride, w_stride

    @property
    def in_channels(self):
        return 3

    @property
    def total_stride(self):
        return self._probe_total_stride()

    @property
    def out_channels(self):
        _, local_features = self._compute_sample_output()
        return local_features.shape[1]

    @property
    def feature_map_spatial_size(self):
        _, local_features = self._compute_sample_output()
        return local_features.shape[-2:]

    def describe(self):
        desc_str = (
            f"{self.__class__.__name__} (with input size={_SAMPLE_INPUT_DIM}):"
            f"\n\tSpatial dimensions of the feature-maps={self.feature_map_spatial_size}"
            f"\n\tNumber of feature_map channels={self.out_channels}"
            f"\n\tTotal stride={self.total_stride}"
        )
        return desc_str
