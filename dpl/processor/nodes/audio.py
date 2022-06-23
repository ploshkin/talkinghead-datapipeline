from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dpl.processor.nodes.base import BaseNode, BaseResource
from dpl.processor.datatype import DataType
import dpl.wav2vec
import dpl.common


class Wav2vecResource(BaseResource):
    def __init__(self, checkpoint: str, device: str) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.device = device

    def load(self) -> None:
        self.encoder = dpl.wav2vec.AudioFeatureExtractor(self.checkpoint, self.device)
        super().load()

    def unload(self) -> None:
        del self.encoder
        super().unload()


class Wav2vecNode(BaseNode):
    input_types = [DataType.WAV]
    output_types = [DataType.WAV2VEC, DataType.VOLUME]

    def __init__(
        self,
        device: str,
        checkpoint: str = "facebook/wav2vec2-base",
        batch_size: int = 8,
        num_workers: int = 4,
        recompute: bool = False,
    ) -> None:
        super().__init__(recompute)
        self.resource = Wav2vecResource(checkpoint, device)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run_sequence(self, start: int, num: int, verbose: bool) -> None:
        name = self.__class__.__name__

        indices = self._choose_indices_to_process(range(start, start + num))
        dataloader = self.make_dataloader(indices)
        sample_rate = dataloader.dataset.sample_rate

        if verbose:
            desc = self.get_description(start, num)
            dataloader = tqdm(dataloader, desc=desc, total=len(dataloader))

        with self.resource:
            data_index = 0
            for batch in dataloader:
                batch_size = len(batch)
                try:
                    outputs = self.resource.encoder.batch_encode(batch, sample_rate)

                except Exception as exc:
                    error_indices = indices[data_index : data_index + batch_size]
                    inputs = {
                        "wav": [self.inputs["wav"][index] for index in error_indices]
                    }
                    self._report.add_error(inputs, str(exc))

                else:
                    # TODO: Parallelize saving?
                    for offset in range(batch_size):
                        index = indices[data_index + offset]
                        for key, value in outputs.items():
                            path = self.outputs[key][index]
                            path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(path, value[offset])

                data_index += batch_size

    def make_dataloader(self, indices: List[int]) -> DataLoader:
        audio_paths = [self.inputs["wav"][index] for index in indices]
        return DataLoader(
            dpl.wav2vec.WavDataset(audio_paths),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def _choose_indices_to_process(self, indices: Iterable[int]) -> List[int]:
        indices_to_process = []
        for index in indices:
            input_paths = {
                dt.key: self.inputs[dt.key][index] for dt in self.input_types
            }
            output_paths = {
                dt.key: self.outputs[dt.key][index] for dt in self.output_types
            }

            input_exists = self.check_exist(input_paths)
            need_computation = self.recompute or not self.check_exist(output_paths)

            if input_exists and need_computation:
                indices_to_process.append(index)

        return indices_to_process
