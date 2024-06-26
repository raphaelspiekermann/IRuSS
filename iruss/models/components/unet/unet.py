import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .attention import SpatialTransformer
from .util import avg_pool_nd, checkpoint, conv_nd, normalization, zero_module


class ContextAwareSequential(nn.Sequential):
    """A sequential module that passes context to each layer."""

    def forward(self, x, context=None):
        for layer in self:
            if isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then upsampling occurs in the
        inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then downsampling occurs in the
        inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial convolution instead of a
        smaller 1x1 convolution to change the channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(
            self._forward, (x,), self.parameters(), True
        )  # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    """A module which performs QKV attention.

    Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different order."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class UNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which attention will take
        place. May be a set, list, or tuple. For example, if this contains 4, then at 4x
        downsampling, attention will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use a fixed channel width
        per attention head.
    :param num_heads_upsample: works with num_heads to set a different number of heads for
        upsampling. Deprecated.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially increased
        efficiency.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,  # must be powers of 2
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        legacy=True,
    ):
        super().__init__()

        # init params
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample if num_heads_upsample != -1 else num_heads

        # private params
        self._dims = dims
        self._resblock_updown = resblock_updown
        self._use_new_attention_order = use_new_attention_order
        self._transformer_depth = transformer_depth
        self._legacy = legacy
        self._use_spatial_transformer = use_spatial_transformer
        self._context_dim = context_dim

        # check params
        self._check_params()

        # build UNet
        input_block_chans, ch, ds = self._make_input_blocks()  # Downsampling phase
        ch = self._make_middle_block(ch)  # Middle block
        ch, ds = self._make_output_blocks(input_block_chans, ch, ds)  # Upsampling phase

        # create output
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(self._dims, self.model_channels, self.out_channels, 3, padding=1)),
        )

    def _check_params(self):
        if self._use_spatial_transformer:
            assert (
                self._context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if self._context_dim is not None:
            assert (
                self._use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."

        if self.num_heads == -1:
            assert (
                self.num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if self.num_head_channels == -1:
            assert self.num_heads != -1, "Either num_heads or num_head_channels has to be set"

        assert not (
            self.num_head_channels != -1 and self.num_heads != -1
        ), "Either num_heads or num_head_channels has to be set, not both"

    def _attention_config(self, ch):
        num_heads = self.num_heads  # TODO: validate this
        if self.num_head_channels == -1:
            dim_head = ch // self.num_heads
        else:
            num_heads = ch // self.num_head_channels
            dim_head = self.num_head_channels
        if self._legacy:
            num_heads = 1  # TODO: validate this
            dim_head = ch // num_heads if self._use_spatial_transformer else self.num_head_channels

        return num_heads, dim_head

    def _make_input_blocks(self):
        self.input_blocks = nn.ModuleList(
            [
                ContextAwareSequential(
                    conv_nd(self._dims, self.in_channels, self.model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = self.model_channels
        input_block_chans = [self.model_channels]
        ch = self.model_channels
        ds = 1  # downsample factor
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.dropout,
                        out_channels=mult * self.model_channels,
                        dims=self._dims,
                        use_checkpoint=self.use_checkpoint,
                    )
                ]
                ch = mult * self.model_channels
                if ds in self.attention_resolutions:
                    num_heads, dim_head = self._attention_config(ch)
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=self._use_new_attention_order,
                            use_checkpoint=self.use_checkpoint,
                        )
                        if not self._use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=self._transformer_depth,
                            context_dim=self._context_dim,
                        )  # defaults to self-attention when no context is given
                    )
                self.input_blocks.append(ContextAwareSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    ContextAwareSequential(
                        ResBlock(
                            ch,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self._dims,
                            down=True,
                            use_checkpoint=self.use_checkpoint,
                        )
                        if self._resblock_updown
                        else Downsample(
                            ch, self.conv_resample, dims=self._dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        return input_block_chans, ch, ds

    def _make_middle_block(self, ch):
        num_heads, dim_head = self._attention_config(ch)

        self.middle_block = ContextAwareSequential(
            ResBlock(
                ch,
                self.dropout,
                dims=self._dims,
                use_checkpoint=self.use_checkpoint,
            ),
            (
                AttentionBlock(
                    ch,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=self._use_new_attention_order,
                    use_checkpoint=self.use_checkpoint,
                )
                if not self._use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=self._transformer_depth,
                    context_dim=self._context_dim,
                )
            ),
            ResBlock(
                ch,
                self.dropout,
                dims=self._dims,
                use_checkpoint=self.use_checkpoint,
            ),
        )
        self._feature_size += ch

        return ch

    def _make_output_blocks(self, input_block_chans, ch, ds):
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.dropout,
                        out_channels=self.model_channels * mult,
                        dims=self._dims,
                        use_checkpoint=self.use_checkpoint,
                    )
                ]
                ch = self.model_channels * mult
                if ds in self.attention_resolutions:
                    num_heads, dim_head = self._attention_config(ch)

                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=self._use_new_attention_order,
                            use_checkpoint=self.use_checkpoint,
                        )
                        if not self._use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=self._transformer_depth,
                            context_dim=self._context_dim,
                        )
                    )
                if level and i == self.num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.dropout,
                            out_channels=out_ch,
                            dims=self._dims,
                            up=True,
                            use_checkpoint=self.use_checkpoint,
                        )
                        if self._resblock_updown
                        else Upsample(ch, self.conv_resample, dims=self._dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(ContextAwareSequential(*layers))
                self._feature_size += ch

        return ch, ds

    def forward(self, x, context=None, **kwargs):
        """Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, context)
            hs.append(h)
        h = self.middle_block(h, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, context)
        h = h.type(x.dtype)

        return self.out(h)
