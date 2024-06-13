import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import torch.utils.data as data


class VisionDataset(data.Dataset):
    """Copied and adjusted from torchvision.datasets.VisionDataset.

    Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.

    Args:
        root (string, optional): Root directory of dataset. Only used for `__repr__`.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive. # noqa: RST304
    """

    _repr_indent = 4

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)

        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""
