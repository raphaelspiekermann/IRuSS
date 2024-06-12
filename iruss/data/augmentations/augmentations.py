import abc
import math
import random
from copy import deepcopy
from typing import List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torchvision import transforms
from typeguard import typechecked

from .conditionings import positional_encoding, sum_index_matrix
from .helpers import _get_interpolation_mode, _setup_size

__all__ = [
    "ImageTensorType",
    "FeatureAugmentationBase",
    "Identity",
    "Resize",
    "ToTensor",
    "Grayscale",
    "GrayscaleToRGB",
    "VerticalFlip",
    "HorizontalFlip",
    "Normalize",
    "MeanGrayscale",
    "GaussianBlur",
    "ColorJitter",
    "RandomBrightness",
    "RandomContrast",
    "RandomChannelDropout",
    "RandomAffine",
    "RandomRotation",
    "RandomMasking",
    "BlockMasking",
    "RandomCrop",
    "RandomGrayscale",
    "RandomChoice",
    "RandomApply",
    "RandomAffine",
    "RandomRotation",
    "RandomMasking",
    "BlockMasking",
    "RandomCrop",
    "RandomGrayscale",
    "RandomChoice",
    "AugmentationPipeline",
    "ContrastiveAugmentationPipeline",
]


# Custom type alias for the input images (either tensors or PIL images)
ImageTensorType: TypeAlias = Union[Tensor, Image.Image]


class FeatureAugmentationBase(nn.Module):
    """Base class for transformations that can be in feature space."""

    @abc.abstractmethod
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        raise NotImplementedError


class Identity(FeatureAugmentationBase):
    """Identity transformation."""

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        return x if mask is None else (x, mask)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Resize(FeatureAugmentationBase):
    @typechecked
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: Union[str, transforms.InterpolationMode] = "nearest",
        max_size: Optional[int] = None,
        antialias: Optional[bool] = True,
    ):
        super().__init__()
        self.size = size
        self.interpolation = _get_interpolation_mode(interpolation)
        self.max_size = max_size
        self.antialias = antialias

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        x = F.resize(x, self.size)
        if mask is None:
            return x
        mask = F.resize(mask, self.size, interpolation=Image.NEAREST)
        return x, mask

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"


class ToTensor(FeatureAugmentationBase):
    def __init__(self):
        super().__init__()

    @typechecked
    def forward(
        self, x: Image.Image, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return F.to_tensor(x) if mask is None else (F.to_tensor(x), mask)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Grayscale(FeatureAugmentationBase):
    def __init__(self):
        super().__init__()

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        input_channel, _, _ = F.get_dimensions(x)

        if isinstance(x, Tensor):
            x = F.rgb_to_grayscale(x)

            if input_channel != 1:
                dim = len(x.shape)
                repetitions = [1] * dim
                repetitions[-3] = input_channel
                x = x.repeat(*repetitions)
        elif isinstance(x, Image.Image):
            x = F.rgb_to_grayscale(x, num_output_channels=input_channel)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

        return x if mask is None else (x, mask)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class GrayscaleToRGB(FeatureAugmentationBase):
    def __init__(self):
        super().__init__()

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = x.repeat(3, 1, 1)
        return x if mask is None else (x, mask)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class VerticalFlip(FeatureAugmentationBase):
    """Vertical flip transformation."""

    def __init__(self):
        super().__init__()

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        return F.vflip(x) if mask is None else (F.vflip(x), F.vflip(mask))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class HorizontalFlip(FeatureAugmentationBase):
    """Horizontal flip transformation."""

    def __init__(self):
        super().__init__()

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        return F.hflip(x) if mask is None else (F.hflip(x), F.hflip(mask))

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Normalize(FeatureAugmentationBase):
    """Normalize transformation."""

    @typechecked
    def __init__(self, mean: List[float], std: List[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = F.normalize(x, self.mean, self.std)
        return x if mask is None else (x, mask)

    @typechecked
    def unnormalize(self, x: Tensor) -> Tensor:
        return F.normalize(
            x, [-m / s for m, s in zip(self.mean, self.std)], [1 / s for s in self.std]
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class MeanGrayscale(FeatureAugmentationBase):
    def __init__(self):
        super().__init__()

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        assert x.ndim == 3, f"Invalid number of dimensions: {x.ndim}"
        ch_dim = 0
        num_channels = x.shape[ch_dim]
        gray_img = torch.mean(x, dim=ch_dim, keepdim=True)
        gray_img = (
            gray_img.repeat(1, num_channels, 1, 1)
            if ch_dim == 1
            else gray_img.repeat(num_channels, 1, 1)
        )

        if mask is None:
            return gray_img
        return gray_img, mask

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class GaussianBlur(FeatureAugmentationBase):
    @typechecked
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int], List[int]],
        sigma: Union[float, Tuple[float, float], List[float]] = (0.1, 2),
    ):
        super().__init__()
        self._kernel_size = kernel_size
        self._sigma = sigma
        self._gaussian_blur = transforms.GaussianBlur(kernel_size, sigma)

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        x = self._gaussian_blur(x)
        return x if mask is None else (x, mask)

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size={self._kernel_size}, sigma={self._sigma})"


class ColorJitter(FeatureAugmentationBase):
    @typechecked
    def __init__(
        self,
        brightness: Union[float, Tuple[float, float]] = 0,
        contrast: Union[float, Tuple[float, float]] = 0,
        saturation: Union[float, Tuple[float, float]] = 0,
        hue: Union[float, Tuple[float, float]] = 0,
    ):
        super().__init__()
        self._color_jitter = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        if mask is None:
            return self._color_jitter(x)
        return self._color_jitter(x), mask

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"\nBrightness={self._color_jitter.brightness},"
            f"\nContrast={self._color_jitter.contrast},"
            f"\nSaturation={self._color_jitter.saturation},"
            f"\nHue={self._color_jitter.hue})"
        )
        return s


class RandomBrightness(FeatureAugmentationBase):
    @typechecked
    def __init__(self, brightness: Union[float, Tuple[float, float]] = 0):
        super().__init__()
        self._color_jitter = transforms.ColorJitter(brightness=brightness)

    @property
    def brightness(self):
        return self._color_jitter.brightness

    @brightness.setter
    @typechecked
    def brightness(self, value: Union[float, Tuple[float, float]]):
        self._color_jitter.brightness = self._color_jitter._check_input(value, "brightness")

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        brightness_factor = transforms.ColorJitter.get_params(self.brightness, None, None, None)[1]
        if mask is None:
            return (brightness_factor * x).clamp(0, x.max())
        return (brightness_factor * x).clamp(0, x.max()), mask

    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self.brightness})"


class RandomContrast(FeatureAugmentationBase):
    @typechecked
    def __init__(self, contrast: Union[float, Tuple[float, float]] = 0):
        super().__init__()
        self._color_jitter = transforms.ColorJitter(contrast=contrast)

    @property
    def contrast(self):
        return self._color_jitter.contrast

    @contrast.setter
    @typechecked
    def contrast(self, value: Union[float, Tuple[float, float]]):
        self._color_jitter.contrast = self._color_jitter._check_input(value, "contrast")

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        contrast_factor = transforms.ColorJitter.get_params(None, self.contrast, None, None)[2]
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        mean = torch.mean(x.to(dtype), dim=(-2, -1), keepdim=True)
        if mask is None:
            return (contrast_factor * x + (1.0 - contrast_factor) * mean).clamp(0, x.max())
        return (contrast_factor * x + (1.0 - contrast_factor) * mean).clamp(0, x.max()), mask

    def __repr__(self):
        return f"{self.__class__.__name__}(contrast={self.contrast})"


class RandomChannelDropout(FeatureAugmentationBase):
    @typechecked
    def __init__(self, num_drop_channels: int = 1):
        super().__init__()
        self.num_drop_channels = num_drop_channels

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        c = x.shape[-3]
        assert self.num_drop_channels <= c
        drop_channels = random.sample(range(c), self.num_drop_channels)

        for channel in drop_channels:
            x[..., channel, :, :] = 0

        return x if mask is None else (x, mask)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_drop_channels={self.num_drop_channels})"


class RandomAffine(FeatureAugmentationBase):
    REFERENCE_SIZE = (1024, 1024)

    @typechecked
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float], List[float]] = 0,
        translate: Optional[Tuple[float, float]] = (0.1, 0.1),
        scale: Optional[Tuple[float, float]] = (0.9, 1.1),
        shear: Optional[Union[float, Tuple[float, float], List[float]]] = 10,
        interpolation: Union[str, transforms.InterpolationMode] = "nearest",
    ):
        super().__init__()

        self.interpolation = _get_interpolation_mode(interpolation)

        # use torchvision's RandomAffine to clean up the parameters
        self.random_affine = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
        )

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        _, height, width = F.get_dimensions(x)

        a, t, sc, sh = self.random_affine.get_params(
            self.random_affine.degrees,
            self.random_affine.translate,
            self.random_affine.scale,
            self.random_affine.shear,
            self.REFERENCE_SIZE,
        )

        img_size = [width, height]

        t = tuple(
            t[i] * img_size[i] // self.REFERENCE_SIZE[i] for i in range(len(img_size))
        )  # scale translations according to image size

        interp = self.interpolation

        if mask is None:
            return F.affine(x, a, t, sc, sh, interp)
        return F.affine(x, a, t, sc, sh, interp), F.affine(mask, a, t, sc, sh, interp)

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"\nDegrees={self.random_affine.degrees},"
            f"\nTranslate={self.random_affine.translate},"
            f"\nScale={self.random_affine.scale},"
            f"\nShear={self.random_affine.shear},"
            f"\nInterpolation={self.interpolation})"
        )
        return s


class RandomRotation(FeatureAugmentationBase):
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float], List[float]],
        interpolation: Union[str, transforms.InterpolationMode] = "nearest",
    ):
        super().__init__()
        self._degrees = _setup_size(degrees, name="degrees", req_sizes=(2,))
        self._interpolation = _get_interpolation_mode(interpolation)

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        deg = self._degrees
        angle = float(torch.empty(1).uniform_(float(deg[0]), float(deg[1])).item())

        if mask is None:
            return F.rotate(x, angle, interpolation=self._interpolation)
        return (
            F.rotate(x, angle, interpolation=self._interpolation),
            F.rotate(mask, angle, interpolation=F.InterpolationMode.NEAREST),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(degrees={self._degrees})"


class RandomMasking(FeatureAugmentationBase):
    @typechecked
    def __init__(self, relative_mask_size: float = 0.2):
        super().__init__()
        if not 0 <= relative_mask_size <= 1:
            raise ValueError("relative_size must be between 0 and 1")
        self._relative_mask_size = relative_mask_size  # relative size of the mask

    def _random_masking(self, x: Tensor, anker_point) -> Tensor:
        assert isinstance(x, Tensor), "Random masking only works with tensors."

        mask = torch.ones_like(x, dtype=x.dtype).to(x.device)
        _, height, width = F.get_dimensions(x)

        center_x = int(anker_point[0] * width)
        center_y = int(anker_point[1] * height)

        mask_h = int(self._relative_mask_size * height)
        mask_w = int(self._relative_mask_size * width)

        height_start = max(0, center_y - mask_h // 2)
        height_end = min(height, center_y + mask_h // 2)

        width_start = max(0, center_x - mask_w // 2)
        width_end = min(width, center_x + mask_w // 2)

        assert x.ndim == 3, f"Invalid number of dimensions: {x.ndim}"
        mask[:, height_start:height_end, width_start:width_end] = 0

        return x * mask

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        DIMENSIONS = 2  # Number of dimensions for the anchor point
        anker_pt = torch.rand(DIMENSIONS)
        if mask is None:
            return self._random_masking(x, anker_pt)
        return self._random_masking(x, anker_pt), self._random_masking(mask, anker_pt)

    def __repr__(self):
        return f"{self.__class__.__name__}(relative_mask_size={self._relative_mask_size})"


class BlockMasking(FeatureAugmentationBase):
    @typechecked
    def __init__(self, relative_mask_size: float = 0.25, block_size: int = 16):
        super().__init__()
        self._relative_mask_size = relative_mask_size
        self._block_size = block_size

    @typechecked
    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        h, w = x.shape[-2:]

        # compute number of blocks
        num_blocks_h = math.ceil(h / self._block_size)
        num_blocks_w = math.ceil(w / self._block_size)

        num_blocks = num_blocks_h * num_blocks_w

        # build blocks
        blocks = []
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                lo_h = i * self._block_size
                hi_h = min(h, (i + 1) * self._block_size)

                lo_w = j * self._block_size
                hi_w = min(w, (j + 1) * self._block_size)

                blocks.append((lo_h, hi_h, lo_w, hi_w))

        assert len(blocks) == num_blocks, f"{len(blocks) = } != {num_blocks = }"

        # select random blocks
        num_blocks_to_mask = round(self._relative_mask_size * num_blocks)
        blocks_to_mask = random.sample(blocks, num_blocks_to_mask)

        # build mask (bool_mask)
        bool_mask = torch.ones(size=(h, w), dtype=torch.bool, device=x.device)
        for lo_h, hi_h, lo_w, hi_w in blocks_to_mask:
            bool_mask[lo_h:hi_h, lo_w:hi_w] = 0

        bool_mask = bool_mask.float()

        # apply mask (bool_mask)
        if mask is None:
            return x * bool_mask

        mask_h, mask_w = x.shape[-2:]
        bool_mask2 = F.resize(
            bool_mask,
            (mask_h, mask_w),
            interpolation=F.InterpolationMode.NEAREST,
            antialias=False,
        )

        return x * bool_mask, mask * bool_mask2

    def __repr__(self):
        return f"{self.__class__.__name__}(relative_mask_size={self._relative_mask_size})"


class RandomCrop(FeatureAugmentationBase):
    """Random crop transformation."""

    SAMPLE_RESOLUTION = (100, 100)  # Uses as reference for the crop

    @typechecked
    def __init__(
        self,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: Union[str, transforms.InterpolationMode] = "nearest",
        antialias: bool = True,
    ):
        super().__init__()
        self._sample_input = torch.zeros(size=(1, 3, *self.SAMPLE_RESOLUTION), dtype=torch.float)
        self._scale = scale
        self._ratio = ratio
        self._interpolation = _get_interpolation_mode(interpolation)
        self._antialias = antialias

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            self._sample_input, self._scale, self._ratio
        )

        assert top >= 0 and left >= 0 and height >= 0 and width >= 0, (
            f"Invalid crop parameters: {top = }, {left = }, " f"{height = }, {width = }"
        )
        assert top + height <= self.SAMPLE_RESOLUTION[0]
        assert left + width <= self.SAMPLE_RESOLUTION[1]

        w, h = F.get_image_size(x)

        # scale to spatial size
        top = int(top * w / self.SAMPLE_RESOLUTION[0])
        left = int(left * h / self.SAMPLE_RESOLUTION[1])
        height = int(height * w / self.SAMPLE_RESOLUTION[0])
        width = int(width * h / self.SAMPLE_RESOLUTION[1])

        params = {
            "top": top,
            "left": left,
            "height": height,
            "width": width,
            "size": (w, h),
            "interpolation": self._interpolation,
            "antialias": self._antialias,
        }

        if mask is None:
            return F.resized_crop(x, **params)
        return F.resized_crop(x, **params), F.resized_crop(mask, **params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self._scale}, ratio={self._ratio})"


class RandomGrayscale(FeatureAugmentationBase):
    @typechecked
    def __init__(self, p: float = 0.5):
        super().__init__()
        self._random_grayscale = RandomApply(Grayscale(), p=p)

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        return self._random_grayscale(x, mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self._random_grayscale._p})"


class RandomChoice(FeatureAugmentationBase):
    """Random choice transformation."""

    @typechecked
    def __init__(self, transforms: List[FeatureAugmentationBase], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrapped_transforms = transforms

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        selected_transform = random.choice(self._wrapped_transforms)  # nosec
        if mask is None:
            return selected_transform(x)
        return selected_transform(x, mask)

    def __repr__(self):
        return self.__class__.__name__ + f"({self._wrapped_transforms})"


class RandomApply(FeatureAugmentationBase):
    @typechecked
    def __init__(self, transform: FeatureAugmentationBase, p: float = 0.5):
        super().__init__()
        assert 0 <= p <= 1
        self._wrapped_transform = transform
        self._p = p

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        if torch.rand(1).item() < self._p:
            return self._wrapped_transform(x, mask)
        return x if mask is None else (x, mask)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"\nTransform={self._wrapped_transform},"
            f"\nProbability={self._p})"
        )
        return s


class Compose(FeatureAugmentationBase):
    """Compose transformations."""

    @typechecked
    def __init__(self, transforms: List[FeatureAugmentationBase]):
        super().__init__()
        self._wrapped_transforms = transforms

    @typechecked
    def forward(
        self, x: ImageTensorType, mask: Optional[Tensor] = None
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        for transform in self._wrapped_transforms:
            if mask is None:
                x = transform(x)
            else:
                x, mask = transform(x, mask)

        return x if mask is None else (x, mask)

    @typechecked
    def __getitem__(self, idx: int):
        return self._wrapped_transforms[idx]

    def __len__(self):
        return len(self._wrapped_transforms)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self._wrapped_transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomOrder(FeatureAugmentationBase):
    @typechecked
    def __init__(self, transforms: Union[Compose, List[FeatureAugmentationBase]]):
        super().__init__()
        self._wrapped_transform = transforms

    @typechecked
    def forward(self, x: ImageTensorType, mask: Optional[Tensor] = None):
        if isinstance(self._wrapped_transform, Compose):
            transforms = self._wrapped_transform._wrapped_transforms
        else:
            transforms = self._wrapped_transform

        order = torch.randperm(len(transforms))
        for idx in order:
            transform = transforms[idx]
            if mask is None:
                x = transform(x)
            else:
                x, mask = transform(x, mask)

        return x if mask is None else (x, mask)


class AugmentationPipeline(FeatureAugmentationBase):
    """
    Args:
        transforms (List[ReversibleTransformation]): List of transformations.
        return_masks (bool, optional): If True, the transformation returns the masks.
            Defaults to False.
        masking_type (str, optional): Type of masking. Defaults to "sum_index_matrix".
        pos_enc_chn (int, optional): Number of channels for the positional encoding.
    """

    @typechecked
    def __init__(
        self,
        transforms: Union[List[FeatureAugmentationBase], Compose],
        return_masks: bool = False,
        masking_type: str = "sum_index_matrix",  # "sum_index_matrix" | "positional_encoding"
        pos_enc_chn: int = 3,  # positional encoding number of channels
    ):
        super().__init__()

        assert masking_type in ["sum_index_matrix", "positional_encoding"], (
            f"Invalid masking type: {masking_type}. "
            "Supported types are 'sum_index_matrix' and 'positional_encoding'."
        )

        _is_compose = isinstance(transforms, Compose)
        self._wrapped_transform = transforms if _is_compose else Compose(transforms)

        self._return_masks = return_masks
        self._masking_type = masking_type
        self._pos_enc_chn = pos_enc_chn

    @typechecked
    def forward(
        self, x: ImageTensorType
    ) -> Union[ImageTensorType, Tuple[ImageTensorType, Tensor]]:
        if isinstance(x, Tensor):
            assert x.ndim == 3, f"Invalid number of dimensions: {x.ndim}"
        else:
            assert isinstance(x, Image.Image)

        if self._return_masks:
            if self._masking_type == "sum_index_matrix":
                w, h = F.get_image_size(x)
                # print(f"{w = }, {h = }")

                # Note: Dimensions are swapped to match the tensor shape,
                #       kinda dumb but yeah...
                masks = sum_index_matrix((h, w))
                masks = masks.float()
                masks /= masks.max()  # normalize to [0, 1]
                masks = masks.unsqueeze(0)  # (C,W,H)
            elif self._masking_type == "positional_encoding":
                masks = positional_encoding(F.get_image_size(x), self._pos_enc_chn)
                masks = masks.float()
                masks /= masks.max()  # normalize to [0, 1]
                masks = masks.transpose(0, 2)  # (H,W,C) -> (C,W,H)
                if x.ndim == 4:
                    masks = masks.unsqueeze(0)  # add batch dimension
                    masks = masks.repeat(x.shape[0], 1, 1, 1)
            else:
                raise ValueError(f"Unsupported masking type: {self._masking_type}")

        if self._return_masks:
            return self._wrapped_transform(x, masks)
        return self._wrapped_transform(x)

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"\nTransforms={self._wrapped_transform},"
            f"\nReturn Masks={self._return_masks},"
            f"\nMasking Type={self._masking_type},"
            f"\nPositional Encoding Channels={self._pos_enc_chn})"
        )
        return s

    # def __str__(self):
    #    return super().__str__() + f"\nTransform: {self._wrapped_transform}"


class ContrastiveAugmentationPipeline(FeatureAugmentationBase):
    """Augmentation pipeline for contrastive learning.

    Args:
        augmentation_pipeline (AugmentationPipeline): Augmentation pipeline that will be cloned for the query and key.
    """

    @typechecked
    def __init__(self, augmentation_pipeline: AugmentationPipeline):
        super().__init__()

        self.q_augmentation_pipeline = augmentation_pipeline
        self.k_augmentation_pipeline = deepcopy(augmentation_pipeline)

    @typechecked
    def forward(
        self, x: ImageTensorType
    ) -> Tuple[
        Union[ImageTensorType, Tuple[ImageTensorType, Tensor]],
        Union[ImageTensorType, Tuple[ImageTensorType, Tensor]],
    ]:
        q = self.q_augmentation_pipeline(x)
        k = self.k_augmentation_pipeline(x)
        return q, k

    def __repr__(self):
        s = (
            f"{self.__class__.__name__}("
            f"\nQuery Augmentation Pipeline={self.q_augmentation_pipeline},"
            f"\nKey Augmentation Pipeline={self.k_augmentation_pipeline})"
        )
        return s
