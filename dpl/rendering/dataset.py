from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class RenderingDataset(Dataset):
    def __init__(
        self,
        verts_path: Path,
        cam_path: Path,
        light_path: Path,
    ) -> None:
        self.verts = np.load(verts_path)
        self.cam = np.load(cam_path)
        self.light = np.load(light_path)
        
        if len(self.verts) != len(self.cam):
            raise RuntimeError("Lengths must be equal")

        if len(self.verts) != len(self.light):
            raise RuntimeError("Lengths must be equal")

        self._length = len(self.verts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "verts": torch.tensor(self.verts[index], dtype=torch.float32),
            "cam": torch.tensor(self.cam[index], dtype=torch.float32),
            "light": torch.tensor(self.light[index], dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self._length
