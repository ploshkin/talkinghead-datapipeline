from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class FlameDataset(Dataset):
    def __init__(
        self,
        shape_path: Path,
        exp_path: Path,
        pose_path: Path,
    ) -> None:
        self.shape = np.load(shape_path)
        self.exp = np.load(exp_path)
        self.pose = np.load(pose_path)
        
        if len(self.shape) != len(self.exp):
            raise RuntimeError("Lengths must be equal")

        if len(self.shape) != len(self.pose):
            raise RuntimeError("Lengths must be equal")

        self._length = len(self.shape)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "shape": torch.tensor(self.shape[index], dtype=torch.float32),
            "exp": torch.tensor(self.exp[index], dtype=torch.float32),
            "pose": torch.tensor(self.pose[index], dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self._length
