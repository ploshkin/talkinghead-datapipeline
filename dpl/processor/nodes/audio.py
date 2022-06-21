from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode, BaseResource, NodeExecReport
from dpl.processor.datatype import DataType
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
    input_types = [DataType.WAV]
    output_types = [DataType.WAV2VEC, DataType.VOLUME]

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

    def run_sequence(self, start: int, num: int, verbose: bool) -> NodeExecReport:
        name = self.__class__.__name__
        report = NodeExecReport(name, start, num)

        dataloader = self.make_dataloader(start, num)
        sample_rate = dataloader.dataset.sample_rate

        if verbose:
            desc = self.get_description(start, num)
            dataloader = tqdm(dataloader, desc=desc, total=len(dataloader))

        with self.resource:
            global_index = start
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

    def make_dataloader(self, start: int, num: int) -> DataLoader:
        return DataLoader(
            dpl.wav2vec.WavDataset(self.inputs["wav"][start : start + num]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda x: x,
        )
