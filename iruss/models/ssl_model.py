import math
from functools import cached_property, partial
from typing import Union

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F  # noqa: N812

from iruss.models.backbones.convnext_v2.convnextv2 import ConvNeXtV2
from iruss.models.components.unet.unet import Upsample as UnetUpsample


class ResNetWrapper(nn.Module):
    def __init__(self, wrapped_module):
        super().__init__()
        self._resnet: torchvision.models.ResNet = wrapped_module
        del self._resnet.avgpool
        del self._resnet.fc

    def forward(self, x):
        x = self._resnet.conv1(x)
        x = self._resnet.bn1(x)
        x = self._resnet.relu(x)
        x = self._resnet.maxpool(x)

        x = self._resnet.layer1(x)
        x = self._resnet.layer2(x)
        x = self._resnet.layer3(x)
        x = self._resnet.layer4(x)

        return x


class ViTWrapper(nn.Module):
    def __init__(self, wrapped_module):
        super().__init__()
        self._vit = wrapped_module
        del self._vit.heads

    def forward(self, x):
        x = self._vit._process_input(x)
        x = self._vit.encoder(x)

        x = x.permute(0, 2, 1)  # B, C, H*W

        b, c = x.shape[:2]
        d = int(math.sqrt(x.shape[-1]))

        assert d * d == x.shape[-1], "Expected square local features"

        return x.reshape(b, c, d, d)


class ConvNeXtV2Wrapper(nn.Module):
    def __init__(self, wrapped_module):
        super().__init__()
        self._convnextv2 = wrapped_module
        del self._convnextv2.head
        del self._convnextv2.norm

    def forward(self, x):
        for i in range(4):
            x = self._convnextv2.downsample_layers[i](x)
            x = self._convnextv2.stages[i](x)

        return x


class SSL_Backbone(nn.Module):
    def __init__(self, wrapped_module):
        super().__init__()
        self._wrapped_module = wrapped_module

    def forward(self, x):
        return self._wrapped_module(x)

    def _probe_in_channels(self):
        ls = []
        with torch.no_grad():
            for ch in range(1, 20):
                sample_input = torch.randn(2, ch, 224, 224)

                device = next(self.parameters()).device
                sample_input = sample_input.to(device)

                try:
                    self.forward(sample_input)
                    ls.append(ch)
                except Exception as e:
                    print(f"Could not determine input channels for {ch} channels: {e}")

        if len(ls) == 1:
            return ls[0]

        raise ValueError(f"Could not determine input channels: {ls}")

    def _probe_out_channels(self):
        with torch.no_grad():
            sample_input = torch.randn(2, self.in_channels, 224, 224)

            device = next(self.parameters()).device
            sample_input = sample_input.to(device)

            out = self.forward(sample_input)

            if out.ndim == 3:
                # Vision Transformer
                return out.shape[2]
            elif out.ndim == 4:
                # Convolutional Encoder
                return out.shape[1]
            else:
                raise ValueError(f"Unsupported output shape: {out.shape}")

    def _probe_total_stride(self):
        with torch.no_grad():
            sample_input = torch.randn(2, self.in_channels, 224, 224)

            device = next(self.parameters()).device
            sample_input = sample_input.to(device)

            sample_output = self.forward(sample_input)

            if sample_output.ndim == 3:
                # print(f"Vision Transformer output shape: {sample_output.shape}")
                # Vision Transformer -> reshape to B, C, H, W

                sample_output = sample_output[:, 1:]  # ignore CLS token
                sample_output = sample_output.permute(0, 2, 1)  # B, C, H*W

                b = sample_output.shape[0]
                c = sample_output.shape[1]
                d = int(math.sqrt(sample_output.shape[-1]))

                assert (
                    d * d == sample_output.shape[-1]
                ), "Expected square local features"

                sample_output = sample_output.reshape(b, c, d, d)

            assert sample_output.ndim == 4, f"Expected 4D output, got {sample_output}"
            # Convolutional Encoder -> dimensions are B, C, H, W
            h_out = sample_output.shape[-2]
            w_out = sample_output.shape[-1]

            h_in = sample_input.shape[-2]
            w_in = sample_input.shape[-1]

            h_stride = h_in // h_out  # there could be a remainder
            w_stride = w_in // w_out

            return h_stride, w_stride

    @staticmethod
    def from_basemodel(basemodel: nn.Module) -> "SSL_Backbone":
        if basemodel.__class__ == torchvision.models.ConvNeXt:
            return SSL_Backbone(basemodel.features)

        if basemodel.__class__ == torchvision.models.ResNet:
            return SSL_Backbone(ResNetWrapper(basemodel))

        if basemodel.__class__ == torchvision.models.VisionTransformer:
            return SSL_Backbone(ViTWrapper(basemodel))

        if basemodel.__class__ == ConvNeXtV2:
            return SSL_Backbone(ConvNeXtV2Wrapper(basemodel))

        raise NotImplementedError(f"Unsupported model: {basemodel}")

    @cached_property
    def in_channels(self):
        if isinstance(self._wrapped_module, ResNetWrapper):
            return self._wrapped_module._resnet.conv1.in_channels
        elif isinstance(self._wrapped_module, ConvNeXtV2Wrapper):
            return self._wrapped_module._convnextv2.downsample_layers[0][0].in_channels

        # Bruteforce probing
        return self._probe_in_channels()

    @cached_property
    def total_stride(self):
        return self._probe_total_stride()

    @cached_property
    def out_channels(self):
        return self._probe_out_channels()


class LocalProjector(nn.Module):
    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        num_upsampling_layers: int = 2,
        use_deconvolution: bool = False,
    ) -> None:
        super().__init__()
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._num_upsampling_layers = num_upsampling_layers
        self._use_deconvolution = use_deconvolution
        self._use_upsampling = num_upsampling_layers > 0

        self.channel_embedding = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )  # 1x1 conv to change channel dim to representation size

        if self._use_upsampling:
            self.upsampling_layer = nn.Sequential(
                *(
                    self.create_upsampling_block(output_channel, use_deconvolution)
                    for _ in range(num_upsampling_layers)
                )
            )

    def forward(self, x):
        x = self.channel_embedding(x)
        if self._use_upsampling:
            x = self.upsampling_layer(x)
        return x

    @staticmethod
    def create_upsampling_block(repr_size: int, use_deconvolution: bool):
        if use_deconvolution:
            return nn.Sequential(
                nn.ConvTranspose2d(repr_size, repr_size, 2, stride=2, padding=0),
                nn.BatchNorm2d(repr_size),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                UnetUpsample(channels=repr_size, use_conv=True, dims=2),
                nn.BatchNorm2d(repr_size),
                nn.ReLU(),
            )

    @property
    def use_deconvolution(self):
        return self._use_deconvolution

    @property
    def num_upsampling_layers(self):
        return self._num_upsampling_layers

    @property
    def input_channel(self):
        return self._input_channel

    @property
    def output_channel(self):
        return self._output_channel


class GlobalProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self._output_dim = output_dim
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim


def import_model_from_module(module_name: str, model_name: str, **kwargs) -> nn.Module:
    import importlib

    module = importlib.import_module(module_name)
    model = getattr(module, model_name)(**kwargs)
    return model


SEARCH_SPACE = {
    "convnextv2": partial(
        import_model_from_module, "src.models.backbones.convnext_v2.convnextv2"
    ),
    "torchvision": partial(import_model_from_module, "torchvision.models.get_model"),
    # "timm": partial(import_model_from_module, "timm.models.vision_transformer"),
}


def get_model_by_name(model_name: str, **kwargs) -> nn.Module:
    for module_name, model_loader in SEARCH_SPACE.items():
        try:
            model = model_loader(model_name, **kwargs)
            return model
        except Exception:
            # model not found using this loader -> try next
            pass  # noqa: E701

    search_locations = ", ".join(SEARCH_SPACE.keys())
    raise ValueError(f"Model not found: {model_name}. Search space: {search_locations}")


class SSL_Model(nn.Module):
    """
    A wrapper around a backbone model that adds a local and global projector.

    Args:
        basemodel (Union[str, nn.Module]): The backbone model. If a string is passed,
            the model is loaded using `torchvision.models.get_model`.
        model_config (dict, optional): The configuration for the model. Defaults to None.
        representation_size (int, optional): The size of the representation. Defaults to 128.
        hidden_size_multiplier (int, optional): The multiplier for the hidden size of the
            global projector. Defaults to 2.
        num_upsampling_layers (int, optional): The number of upsampling layers in the local
            projector. Defaults to 2.
        use_deconvolution (bool, optional): Whether to use deconvolution in the local projector.
            Defaults to False.
    """

    def __init__(
        self,
        basemodel: Union[str, nn.Module],
        model_config: dict = None,
        representation_size: int = 128,
        hidden_size_multiplier: int = 2,
        num_upsampling_layers: int = 2,
        use_deconvolution: bool = False,
    ):
        super().__init__()

        model_config = model_config or {}

        if isinstance(basemodel, str):
            basemodel = get_model_by_name(basemodel, **model_config)
        else:
            assert isinstance(
                basemodel, nn.Module
            ), f"Unsupported basemodel: {basemodel}"
            assert model_config == {}, "Model config is not supported for custom models"

        self.backbone = SSL_Backbone.from_basemodel(basemodel)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.representation_size = representation_size

        self.backbone_out_channels = self.backbone.out_channels

        self.local_projector = LocalProjector(
            input_channel=self.backbone_out_channels,
            output_channel=representation_size,
            num_upsampling_layers=num_upsampling_layers,
            use_deconvolution=use_deconvolution,
        )

        self.global_projector = GlobalProjector(
            input_dim=self.backbone_out_channels,
            hidden_dim=self.backbone_out_channels * hidden_size_multiplier,
            output_dim=representation_size,
        )

    @property
    def use_deconvolution(self):
        return isinstance(self.local_projector[1], nn.ConvTranspose2d)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _forward_local_projector(self, x: torch.Tensor) -> torch.Tensor:
        return self.local_projector(x)

    def _forward_global_projector(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.global_projector(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_backbone_features: bool = False,
        return_local_features: bool = False,
        return_global_features: bool = False,
    ) -> Union[torch.Tensor, dict]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_local_features (bool, optional): Whether to return local features.
                Defaults to False.
            return_global_features (bool, optional): Whether to return global features.
                Defaults to False.

        Returns:
            Union[torch.Tensor, dict]: If neither `return_local_features` nor
                `return_global_features` is True, returns the representation tensor.
                Otherwise, returns a dictionary with the following keys:
                - `representation`: The representation tensor.
                - 'backbone_features': The output of the backbone.
                - `local_features`: The local features tensor (after local projection).
                - `global_features`: The global features tensor (after global projection).
        """
        backbone_out = self._forward_backbone(x)
        representation = self.flatten(self.avg_pool(backbone_out))

        if not any(
            [return_backbone_features, return_local_features, return_global_features]
        ):
            return representation  # default
        else:
            return_dict = {"representation": representation}

            if return_backbone_features:
                return_dict["backbone_features"] = backbone_out

            if return_local_features:
                return_dict["local_features"] = self._forward_local_projector(
                    backbone_out
                )
            if return_global_features:
                return_dict["global_features"] = self._forward_global_projector(
                    backbone_out
                )

            return return_dict
