import math
from typing import Any, Optional, Union

import torch
import torchvision.models as tv_models
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param

from .backbone import Backbone
from .registry import register_model

_CONFIGS = {
    "vit_b_16_no_pt": {
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "weights": None,
    },
    "vit_b_16_im1k": {
        "patch_size": 16,
        "num_layers": 12,
        "num_heads": 12,
        "hidden_dim": 768,
        "mlp_dim": 3072,
        "weights": tv_models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1,
    },
}


class VitBackbone(tv_models.VisionTransformer, Backbone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, x: Tensor, return_feature_maps: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x_global = x[:, 0]

        if return_feature_maps:
            x_local = x[:, 1:]

            x_local = x_local.permute(0, 2, 1)  # B, C, H*W

            b, c = x_local.shape[:2]
            d = int(math.sqrt(x_local.shape[-1]))

            assert d * d == x_local.shape[-1], "Expected square local features"

            x_local = x_local.reshape(b, c, d, d)

            return x_global, x_local

        return x_global


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights: Optional[WeightsEnum],
    progress: bool = True,
    **kwargs: Any,
) -> VitBackbone:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    vit = VitBackbone(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        vit.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    del vit.heads

    return vit


def _closure(func, *args, **kwargs):
    def _f():
        return func(*args, **kwargs)

    return _f


for rn_name, rn_cfg in _CONFIGS.items():
    register_model(_closure(_vision_transformer, **rn_cfg), name=rn_name)
