from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode, BaseResource, NodeExecReport
import dpl.wav2vec
import dpl.common


class Wav2vecResource(BaseResource):
    def __init__(self, checkpoint: str, device: str) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.reset()

    def __enter__(self) -> "Wav2vecResource":
        self.encoder = dpl.wav2vec.AudioFeatureExtractor(self.checkpoint, self.device)
        return self

    def reset(self) -> None:
        if hasattr(self, "model"):
            del self.encoder
        self.encoder = None


class Wav2vecNode(BaseNode):
    input_keys = ["wav"]
    output_keys = ["wav2vec", "volume"]

    def __init__(
        self,
        device: str,
        checkpoint: str = "facebook/wav2vec2-base",
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.resource = Wav2vecResource(checkpoint, device)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, verbose: bool = False) -> NodeExecReport:
        if not self.is_initialized():
            raise RuntimeError("Inputs and outputs are not specified yet.")

        name = self.__class__.__name__
        report = NodeExecReport(name, total=len(self))

        dataloader = self.make_dataloader()
        sample_rate = dataloader.dataset.sample_rate

        if verbose:
            dataloader = tqdm(dataloader, desc=name, total=len(self))

        with self.resource:
            global_index = 0
            for batch in dataloader:
                batch_size = len(batch)
                try:
                    outputs = self.resource.encoder.batch_encode(batch, sample_rate)

                except Exception as exc:
                    slc = slice(global_index, global_index + batch_size)
                    inputs = {"wav": self.inputs["wav"][slc]}
                    report.add_error(inputs, str(exc))

                else:
                    # TODO: Parallelize saving?
                    for offset in range(batch_size):
                        index = global_index + offset
                        for key, value in outputs.items():
                            path = self.outputs[key][index]
                            path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(path, value[offset])

                global_index += batch_size

        return report

    def make_dataloader(self) -> DataLoader:
        return DataLoader(
            dpl.wav2vec.WavDataset(self.inputs["wav"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
