from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from dpl.common import ImageFolderDataset


class EmocaDataset(ImageFolderDataset):

    SIZE_HW = (224, 224)

    def __init__(
        self,
        images_dir: Path,
        bboxes_path: Path,
        ext: str = ".jpg",
        extend_factor: float = 0.2,
    ) -> None:
        super().__init__(images_dir, ext)
        self.bboxes = np.load(bboxes_path)

        if len(self) != len(self.bboxes):
            raise RuntimeError("Lengths must be equal")

        if np.any(np.isnan(self.bboxes)):
            raise RuntimeError(f"NaN values in bboxes, source = '{bboxes_path}'.")

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index])
        image_array = np.array(
            image.crop(self.bboxes[index][:4]).resize(
                self.SIZE_HW, Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
        return torch.tensor(image_array.transpose(2, 0, 1) / 255.0)
