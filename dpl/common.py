from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import skimage.io as io
import torch
from torch.utils.data import Dataset


def listdir(path: Path, ext: Optional[List[str]] = None) -> List[Path]:
    if ext:
        return sorted((p for p in path.iterdir() if p.suffix in ext))
    return sorted(path.iterdir())


class ImageFolderDataset(Dataset):
    def __init__(
        self, images_dir: Path, ext: str = ".jpg", normalize: bool = False,
    ) -> None:
        super().__init__()
        self.paths = listdir(images_dir, [ext])
        self.normalize = normalize

    def __getitem__(self, index: int) -> torch.Tensor:
        image = io.imread(self.paths[index])
        if self.normalize:
            image = image.astype(np.float32) / 255.
        return torch.tensor(image.transpose(2, 0, 1))

    def __len__(self) -> int:
        return len(self.paths)


class NumpyDataset(Dataset):
    def __init__(self, input_paths: Dict[str, Path]) -> None:
        if not input_paths:
            raise RuntimeError("No paths to numpy arrays specified")

        self.arrays = {key: np.load(path) for key, path in input_paths.items()}

        main_key = list(input_paths.keys())[0]
        length = len(self.arrays[main_key])

        for key, array in self.arrays.items():
            if len(array) != length:
                raise RuntimeError(
                    f"Lengths must be equal: ('{main_key}', '{key}')"
                )

        self._length = length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(array[index], dtype=torch.float32)
            for key, array in self.arrays.items()
        }

    def __len__(self) -> int:
        return self._length
