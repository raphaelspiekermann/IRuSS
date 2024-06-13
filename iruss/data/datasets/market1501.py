import glob
import logging
import re
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from .base import VisionDataset
except ImportError:
    from base import VisionDataset


class Market1501(VisionDataset):
    """Market1501.

    Frequently Asked Questions
    1. What are images beginning with "0000" and "-1"?
    Answer: Names beginning with "0000" are distractors produced by DPM false detection.
    Names beginning with "-1" are junks that are neither good nor bad DPM bboxes.
    So "0000" will have a negative impact on accuracy, while "-1" will have no impact.
    During testing, we rank all the images in "bounding_box_test". Then, the junk images are just neglected; distractor images are not neglected.

    1. "bounding_box_test" file. This file contains 19732 images for testing.
    2. "bounding_box_train" file. This file contains 12936 images for training.
    3. "query" file. It contains 3368 query images. Search is performed in the "bounding_box_test" file.
    4. "gt_bbox" file. It contains 25259 images, all hand-drawn. The images correspond to all the 1501 individuals in the test and training set.It is used to distinguish "good" "junk" and "distractors".
    5. "gt_query" file. For each of the 3368 queries, there are both good and junk relevant images (containing the same individual). This file contains the image index of the good and junk images. It is used during performance evaluation
    """

    DOWNLOAD_URL = "https://tu-dortmund.sciebo.de/s/AFt5VjmbBobjZY5/download"
    DIRECTORY_NAME = "Market-1501-v15.09.15"

    def __init__(
        self,
        root: Union[str, Path] = "data",
        split: Literal["train", "gallery", "query"] = "train",
        use_distractors: bool = False,
        use_junk: bool = False,
        use_market1501_500k: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=root, transform=transform)
        assert self.dataset_dir.exists(), f"Dataset not found at {self.dataset_dir}"

        self.split = split
        self.use_distractors = use_distractors
        self.use_junk = use_junk
        self.use_market1501_500k = use_market1501_500k
        self.index = self._load_index()

    @property
    def dataset_dir(self) -> Path:
        return self.root / self.DIRECTORY_NAME

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "bounding_box_train"

    @property
    def gallery_dir(self) -> Path:
        return self.dataset_dir / "bounding_box_test"

    @property
    def query_dir(self) -> Path:
        return self.dataset_dir / "query"

    @property
    def distractor_dir(self) -> Path:
        return self.dataset_dir / "images"

    @property
    def index_path(self) -> Path:
        return self.dataset_dir / "index.csv"

    def _build_index(self):
        cols = [
            "split",
            "image_type",
            "image_path",
            "pid",
            "camera_id",
            "market1501_500k",
        ]

        dirs = [self.train_dir, self.gallery_dir, self.query_dir, self.distractor_dir]
        splits = ["train", "gallery", "query", "gallery"]

        data = []

        def id_to_img_type(_id: int) -> str:
            if _id == -1:
                return "junk"
            elif _id == 0:
                return "distractor"
            else:
                return "good"

        for dir, split in tqdm(zip(dirs, splits)):
            img_paths = glob.glob((dir / "*.jpg").as_posix())
            pattern = re.compile(r"([-\d]+)_c(\d)")
            market_1501_500k = dir == self.distractor_dir

            for img_path in img_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                assert -1 <= pid <= 1501  # pid == 0 means background
                assert 1 <= camid <= 6
                img_type = id_to_img_type(pid)
                camid -= 1  # index starts from 0

                data.append([split, img_type, img_path, pid, camid, market_1501_500k])

        df = pd.DataFrame(data, columns=cols)
        df.to_csv(self.index_path, index=False)

    def rebuild_index(self):
        self._build_index()
        self.index = self._load_index()

    def _load_index(self) -> dict:
        logging.info(f"Loading index from {self.index_path}")

        if not self.index_path.exists():
            logging.info(f"Index not found at {self.index_path}. Building index...")
            self._build_index()

        df = pd.read_csv(self.index_path)

        query = "split == @self.split"
        if not self.use_distractors:
            query += " and image_type != 'distractor'"

        if not self.use_junk:
            query += " and image_type != 'junk'"

        if not self.use_market1501_500k:
            query += " and market1501_500k == False"

        df_filtered = df.query(query)

        _index = dict()
        for idx, (_, row) in enumerate(df_filtered.iterrows()):
            _index[idx] = {
                "image_path": row["image_path"],
                "pid": row["pid"],
                "camera_id": row["camera_id"],
            }

        return _index

    def __len__(self) -> int:
        return len(self.index)

    def _read_image(self, img_path: Path) -> Image.Image:
        return Image.open(img_path)

    def __getitem__(self, index: int) -> dict:
        # row = self.index.iloc[index]
        row = self.index[index]
        img_path = Path(row["image_path"])
        img = self._read_image(img_path)
        pid = row["pid"]
        # camid = row["camera_id"] # Could be useful later

        if self.transform is not None:
            img = self.transform(img)

        return img, pid


def download_market1501(
    root: Union[str, Path],
    force_redownload=False,
    remove_finished: bool = False,
):
    from torchvision.datasets.utils import download_and_extract_archive

    if isinstance(root, str):
        root = Path(root)

    if not root.exists():
        root.mkdir(parents=True)

    url = Market1501.DOWNLOAD_URL
    filename = Market1501.DIRECTORY_NAME + ".zip"

    if not force_redownload:
        if (root / Market1501.DIRECTORY_NAME).exists():
            logging.info("Market1501-Dataset already downloaded")
            return

    download_and_extract_archive(
        url,
        root,
        filename=filename,
        remove_finished=remove_finished,
    )


if __name__ == "__main__":
    download_market1501("E:/test_dwnld")

    marked_dataset = Market1501(
        root="E:/",
        split="gallery",
        use_distractors=False,
        use_junk=True,
        use_market1501_500k=True,
    )
