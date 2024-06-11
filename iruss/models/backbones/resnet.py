from typing import Any, Optional, Type, Union

import torch
import torchvision.models as tv_models
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import BasicBlock, Bottleneck

from .backbone import Backbone
from .registry import register_model

_CONFIGS = {
    "resnet18_no_pt": {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "weights": None,
    },
    "resnet18_im1k": {
        "block": BasicBlock,
        "layers": [2, 2, 2, 2],
        "weights": tv_models.ResNet18_Weights.IMAGENET1K_V1,
    },
    "resnet50_no_pt": {
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "weights": None,
    },
    "resnet50_im1k": {
        "block": Bottleneck,
        "layers": [3, 4, 6, 3],
        "weights": tv_models.ResNet50_Weights.IMAGENET1K_V2,
    },
}


class ResNetBackbone(tv_models.ResNet, Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: Tensor, return_feature_maps: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_local = self.layer4(x)

        x_global = self.avgpool(x_local)
        x_global = torch.flatten(x_global, 1)

        if return_feature_maps:
            return x_global, x_local

        return x_global


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: list[int],
    weights: Optional[WeightsEnum],
    progress: bool = True,
    **kwargs: Any,
) -> ResNetBackbone:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    resnet = ResNetBackbone(block, layers, **kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        resnet.load_state_dict(state_dict)

    del resnet.fc

    return resnet


def _closure(func, *args, **kwargs):
    def _f():
        return func(*args, **kwargs)

    return _f


for rn_name, rn_cfg in _CONFIGS.items():
    register_model(_closure(_resnet, **rn_cfg), name=rn_name)
