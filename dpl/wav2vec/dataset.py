from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from dpl import common


class WavDataset(Dataset):

    SAMPLE_RATE: int = 16000
    EXT: str = ".wav"

    def __init__(self, paths: List[Path], sample_rate: Optional[int] = None) -> None:
        self.paths = paths
        self.sample_rate = sample_rate or WavDataset.SAMPLE_RATE

    def __getitem__(self, index: int) -> np.ndarray:
        path = self.paths[index]
        waveform, sr = librosa.load(path, sr=None, mono=True)
        if waveform.size:
            waveform = librosa.resample(
                waveform, orig_sr=sr, target_sr=self.sample_rate
            )
        return waveform

    def __len__(self) -> int:
        return len(self.paths)
