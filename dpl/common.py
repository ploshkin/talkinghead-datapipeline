import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import skimage.io as io
import skimage.transform as transform
import torch
from torch.utils.data import Dataset


def listdir(
    path: Path, ext: Optional[List[str]] = None, recursive: bool = False
) -> List[Path]:
    if recursive:
        paths = []
        for current_dir, _, files in os.walk(path):
            for file in files:
                if any(file.endswith(ext_) for ext_ in ext):
                    paths.append(Path(current_dir, file))
        return sorted(paths)

    if ext is not None:
        return sorted(p for p in path.iterdir() if p.suffix in ext)

    return sorted(path.iterdir())


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        ext: str = ".jpg",
        size_hw: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.paths = listdir(images_dir, [ext])
        self.size_hw = size_hw
        self.normalize = normalize

    def __getitem__(self, index: int) -> torch.Tensor:
        image = io.imread(self.paths[index]).astype(np.float32)
        if self.size_hw is not None:
            h, w = image.shape[:2]
            if (h, w) != self.size_hw:
                image = transform.resize(
                    image,
                    self.size_hw,
                    anti_aliasing=True,
                    preserve_range=True,
                )

        if self.normalize:
            image /= 255.0

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
                raise RuntimeError(f"Lengths must be equal: ('{main_key}', '{key}')")

        self._length = length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            key: torch.tensor(array[index], dtype=torch.float32)
            for key, array in self.arrays.items()
        }

    def __len__(self) -> int:
        return self._length
