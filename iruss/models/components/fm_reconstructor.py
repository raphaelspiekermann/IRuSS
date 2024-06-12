from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

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
                self.num_channels + self.num_masks * self.num_mask_channels * self.num_mask_repeats
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

        _, h, w = TF.get_dimensions(x)

        masks = [
            TF.resize(mask, size=(h, w), interpolation=TF.InterpolationMode.NEAREST)
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


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.

    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


# TODO
def cosine_dist(x, y):
    return 1 - F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)


# TODO
def euclidean_dist(x, y):
    return torch.pow(x.unsqueeze(1) - y.unsqueeze(0), 2).sum(2)


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""
    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float("Inf"):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

    return loss
