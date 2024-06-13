from typing import Any, Callable, Optional

import lightning as L
from torch.utils.data import DataLoader, Dataset

from iruss.data.augmentations import augmentations

from .datasets.market1501 import Market1501, download_market1501

default_transforms = augmentations.Compose(
    [
        augmentations.Resize(224, interpolation="bilinear"),
        augmentations.RandomApply(augmentations.HorizontalFlip(), p=0.5),
        augmentations.ToTensor(),
        augmentations.Normalize(mean=[0.481, 0.481, 0.481], std=[0.174, 0.174, 0.174]),
    ]
)


class Market1501DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        use_junk: bool = False,
        use_distractors: bool = False,
        use_market1501_500k: bool = False,
        train_transforms: Optional[Callable] = default_transforms,
        test_transforms: Optional[Callable] = default_transforms,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.data_train: Optional[Dataset] = None
        self.data_gallery: Optional[Dataset] = None
        self.data_query: Optional[Dataset] = None

    def prepare_data(self) -> None:
        download_market1501(self.hparams.data_dir)

    def setup(self, stage=None):
        def _build_market1501_data(split, transform):
            return Market1501(
                root=self.hparams.data_dir,
                split=split,
                use_junk=self.hparams.use_junk,
                use_distractors=self.hparams.use_distractors,
                use_market1501_500k=self.hparams.use_market1501_500k,
                transform=transform,
            )

        self.data_train = _build_market1501_data("train", self.train_transforms)
        self.data_gallery = _build_market1501_data("gallery", self.test_transforms)
        self.data_query = _build_market1501_data("query", self.test_transforms)

    def _dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=self.hparams.drop_last,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.data_train, shuffle=True)

    # def val_dataloader(self) -> DataLoader[Any]:
    #    return self._dataloader(self.data_gallery, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._dataloader(self.data_query, shuffle=False)
