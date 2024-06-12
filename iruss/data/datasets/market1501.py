import glob
import re
from pathlib import Path
from typing import Union

import torch.utils.data as data
from torchvision.datasets import VisionDataset


class Market1501(data.Dataset):
    _junk_pids = [0, -1]
    DOWNLOAD_URL = "https://tu-dortmund.sciebo.de/s/6tNjIqZIVtyi552/download"

    def __init__(self, root: Union[str, Path] = "data", market1501_500k=False):
        super().__init__()
        self.root = root
        self.market1501_500k = market1501_500k

    @property
    def dataset_dir(self) -> Path:
        if self.market1501_500k:
            raise NotImplementedError
        return self.root / "Market-1501-v15.09.15"

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "bounding_box_train"

    @property
    def query_dir(self) -> Path:
        return self.dataset_dir / "query"

    @property
    def gallery_dir(self) -> Path:
        return self.dataset_dir / "bounding_box_test"

    @property
    def extra_gallery_dir(self) -> Path:
        return self.dataset_dir / "images"

    def process_dir(self, dir_path, is_train=True):
        img_paths = ""  # glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
