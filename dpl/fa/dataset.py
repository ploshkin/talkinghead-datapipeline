from pathlib import Path

import numpy as np
import skimage.io as io
import torch
from torch.utils.data import Dataset

from dpl.common import listdir


class FaceAlignmentDataset(Dataset):
    def __init__(self, images_dir: Path, ext: str = ".jpg") -> None:
        self.paths = listdir(images_dir, [ext])

    def __getitem__(self, index: int) -> torch.Tensor:
        image = io.imread(self.paths[index])
        return torch.tensor(image.transpose(2, 0, 1))

    def __len__(self) -> int:
        return len(self.paths)
