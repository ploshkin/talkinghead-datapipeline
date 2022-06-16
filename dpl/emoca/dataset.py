from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from dpl.common import listdir
from dpl.emoca.utils import to_integer, to_square, pad_bbox


class EmocaDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        landmarks_path: Path,
        bboxes_path: Path,
        ext: str = ".jpg",
        size_hw: Tuple[int, int] = (224, 224),
        extend_factor: float = 0.2,
    ) -> None:
        self.paths = listdir(images_dir, [ext])
        self.size_hw = size_hw

        landmarks = np.load(landmarks_path)
        bboxes = np.load(bboxes_path)
        
        if len(self) != len(landmarks):
            raise RuntimeError("Lengths must be equal")

        if len(self) != len(bboxes):
            raise RuntimeError("Lengths must be equal")

        if np.any(np.isnan(landmarks)) or np.any(np.isnan(bboxes)):
            raise RuntimeError("NaN values in EmocaDataset.")

        self.bboxes, self.landmarks = self._preprocess_bboxes_landmarks(
            bboxes, landmarks, extend_factor,
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index])
        image_array = np.array(
            image
            .crop(self.bboxes[index][: 4])
            .resize(self.size_hw, Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        return torch.tensor(image_array.transpose(2, 0, 1) / 255.)

    def __len__(self) -> int:
        return len(self.paths)

    def _preprocess_bboxes_landmarks(
        self, 
        bboxes: np.ndarray,
        landmarks: np.ndarray,
        pad: float, 
        dtype: np.dtype = np.int64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        processed_bboxes = np.empty(bboxes.shape, dtype=dtype)
        processed_landmarks = np.empty(landmarks.shape, dtype=dtype)

        for index, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
            bbox = to_integer(pad_bbox(to_square(bbox), pad), dtype=dtype)
            processed_bboxes[index] = bbox

            lm = lm.copy().astype(dtype)
            lm[:, 0] -= bbox[0]
            lm[:, 1] -= bbox[1]
            processed_landmarks[index] = lm

        return processed_bboxes, processed_landmarks
