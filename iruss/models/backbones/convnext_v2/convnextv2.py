# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.hub import load_state_dict_from_url

from ..backbone import Backbone
from ..registry import register_model
from .drop_layer import DropPath
from .utils import GRN, LayerNorm, load_state_dict, remap_checkpoint_keys
from .weight_init import trunc_normal_


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


_CONFIGS = {
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768]},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024]},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536]},
    "huge": {"depths": [3, 3, 27, 3], "dims": [352, 704, 1408, 2816]},
}

_WEIGHTS = {}

for _version in _CONFIGS:
    version_specific_weights = {
        "pt_only": f"https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_{_version}_1k_224_fcmae.pt",
        "im1k": f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_{_version}_1k_224_ema.pt",
        "im22k": f"https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_{_version}_22k_224_ema.pt",
    }
    _WEIGHTS[_version] = version_specific_weights


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


class ConvNeXtV2Backbone(Backbone):
    """ConvNeXt V2

    Args:
        in_channels (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_channels=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, x: Tensor, return_feature_maps: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x_local = x
        x_global = self.norm(x_local.mean([-2, -1]))

        if return_feature_maps:
            return x_global, x_local

        return x_global


def _convnext(
    config: Optional[Literal["tiny", "base", "large", "huge"]] = None,
    weights: Optional[Literal["pt_only", "im1k", "im22k"]] = None,
    **kwargs,
) -> ConvNeXtV2Backbone:
    # init kwargs for convnext
    config = config or "base"
    config = _CONFIGS[config]
    for k, v in kwargs:
        config[k] = v

    convnext = ConvNeXtV2Backbone(**config)

    if weights:
        weights_url = _WEIGHTS[config][weights]
        device = convnext.parameters().__next__().device

        state_dict = load_state_dict_from_url(weights_url, map_location=device)["model"]
        state_dict = remap_checkpoint_keys(state_dict)
        load_state_dict(convnext, state_dict)

    return convnext


def _closure(func, *args, **kwargs):
    def _f():
        return func(*args, **kwargs)

    return _f


_format_str = "convnext_{}_{}"
for cfg in ["tiny", "base", "large", "huge"]:
    register_model(
        _closure(_convnext, config=cfg, weights=None),
        name=_format_str.format(cfg, "no_pt"),
    )

    # add pretrained models
    for weights in ["pt_only", "im1k", "im22k"]:
        register_model(
            _closure(_convnext, config=cfg, weights=weights),
            name=_format_str.format(cfg, weights),
        )
