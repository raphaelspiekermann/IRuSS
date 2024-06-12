import torch.nn as nn
from torch.nn import functional as F  # noqa: N812

from iruss.models.components.unet import Upsample as UnetUpsample


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
