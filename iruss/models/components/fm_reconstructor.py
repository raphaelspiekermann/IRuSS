from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from .unet import UNetModel


class FM_Reconstructor(nn.Module):
    def __init__(
        self,
        num_channels: int,
        context_per_cross_attention: bool = False,
        num_masks=0,
        num_mask_channels=1,
        num_mask_repeats: int = 1,
        normalize_masks: bool = False,
        attention_resolutions: Tuple[int, ...] = (2, 4),
        channel_mult: Tuple[int, ...] = (1, 2),
        num_heads: int = 4,
        context_dim=None,
        use_InvUnet: bool = False,
    ) -> None:
        super().__init__()

        if use_InvUnet:
            raise NotImplementedError

        assert num_masks in [0, 1, 2]  # 1 = q-mask, 2 = q-mask & k-mask

        # set some defaults, could be passed as parameters later
        self.num_res_blocks = 2
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        self.num_channels = num_channels
        self.context_per_attention = context_per_cross_attention

        if self.context_per_attention:
            self.context_dim = context_dim
            in_channels = self.num_channels
            raise NotImplementedError  # TODO: implement this
        else:
            self.num_masks = num_masks
            self.num_mask_channels = num_mask_channels
            self.num_mask_repeats = num_mask_repeats
            self.normalize_masks = normalize_masks

            in_channels = (
                self.num_channels
                + self.num_masks * self.num_mask_channels * self.num_mask_repeats
            )

        self.unet = UNetModel(
            in_channels=in_channels,
            model_channels=self.num_channels,
            out_channels=self.num_channels,
            num_res_blocks=self.num_res_blocks,
            attention_resolutions=self.attention_resolutions,
            channel_mult=self.channel_mult,
            num_heads=self.num_heads,
            use_spatial_transformer=self.context_per_attention,
        )

    def forward(self, x, masks: Optional[List[torch.Tensor]] = None):
        """
        Args:
            x (torch.Tensor): The feature maps to reconstruct.
            masks (List[torch.Tensor]): The masks that were applied to the original images.
        """
        if masks is None:
            masks = []

        assert len(masks) == self.num_masks
        if self.num_masks == 0:
            return self.unet(x)

        _, h, w = F.get_dimensions(x)

        masks = [
            F.resize(mask, size=(h, w), interpolation=F.InterpolationMode.NEAREST)
            for mask in masks
        ]

        masks_processed = []

        for mask in masks:
            assert (
                mask.shape[0] == x.shape[0]
            ), f"Batch size mismatch: {mask.shape[0]} != {x.shape[0]}"

            if self.normalize_masks:
                # normalize to [0, 1]
                _min = mask.reshape(mask.shape[0], -1).min(dim=1)[0]  # (B,)
                _max = mask.reshape(mask.shape[0], -1).max(dim=1)[0]  # (B,)
                mask = (mask - _min[:, None, None, None]) / (
                    _max[:, None, None, None] - _min[:, None, None, None]
                )  # (B, 1, H, W)

            # repeat the mask
            mask = torch.cat([mask] * self.num_mask_repeats, dim=1)
            masks_processed.append(mask)

        masks_processed = torch.cat(masks_processed, dim=1)

        x = torch.cat([x, masks_processed], dim=1)

        return self.unet(x)
