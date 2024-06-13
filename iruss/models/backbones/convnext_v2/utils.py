# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def remap_checkpoint_keys(ckpt):
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("encoder"):
            k = ".".join(k.split(".")[1:])  # remove encoder in the name
        if k.endswith("kernel"):
            k = ".".join(k.split(".")[:-1])  # remove kernel in the name
            new_k = k + ".weight"
            if len(v.shape) == 3:  # resahpe standard convolution
                kv, in_dim, out_dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = (
                    v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                )
            elif len(v.shape) == 2:  # reshape depthwise convolution
                kv, dim = v.shape
                ks = int(math.sqrt(kv))
                new_ckpt[new_k] = v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
            continue
        elif "ln" in k or "linear" in k:
            k = k.split(".")
            k.pop(-2)  # remove ln and linear in the name
            new_k = ".".join(k)
        else:
            new_k = k
        new_ckpt[new_k] = v

    # reshape grn affine parameters and biases
    for k, v in new_ckpt.items():
        if k.endswith("bias") and len(v.shape) != 1:
            new_ckpt[k] = v.reshape(-1)
        elif "grn" in k:
            new_ckpt[k] = v.unsqueeze(0).unsqueeze(1)
    return new_ckpt


def load_state_dict(model, state_dict, prefix="", ignore_missing="relative_position_index"):
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
