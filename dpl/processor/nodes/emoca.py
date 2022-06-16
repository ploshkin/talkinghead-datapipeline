import collections
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import dpl.emoca
import dpl.common
from dpl.processor.nodes.base import BaseNode, BaseResource


class EmocaResource(BaseResource):
    def __init__(self, weights_path: Path, device: str) -> None:
        self.weights_path = weights_path
        self.device = torch.device(device)
        self.reset()

    def __enter__(self) -> 'EmocaResource':
        model = dpl.emoca.EmocaInferenceWrapper()
        model.load_state_dict(torch.load(self.weights_path))
        self.model = model.to(self.device)
        self.model.eval()
        return self

    def reset(self) -> None:
        if hasattr(self, "model"):
            del self.model
        self.model = None


class EmocaNode(BaseNode):
    input_keys = ["crops"]
    output_keys = ["shape", "tex", "exp", "pose", "cam", "light"]

    def __init__(
        self,
        weights_path: Path,
        device: str,
        batch_size: int = 4,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.resource = EmocaResource(weights_path, device)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run_single(
        self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    ) -> None:
        outputs = self.estimate_flame_codes(input_paths)
        self.save_outputs(outputs, output_paths)

    def estimate_flame_codes(self, input_paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
        keys = list(self.outputs.keys())

        batched_codes = collections.defaultdict(list)
        dataloader = self.make_dataloader(input_paths)
        for index, batch in enumerate(dataloader):
            codes = self.resource.model.encode(batch.to(self.resource.device))
            codes = {
                key: tensor.detach().cpu().numpy()
                for key, tensor in codes.items()
            }
            for key in keys:
                batched_codes[key].append(codes[key])

        return {
            key: np.concatenate(batched_codes[key])
            for key in keys
        }

    def save_outputs(self, outputs: Dict[str, np.ndarray], paths: Dict[str, Path]) -> None:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, outputs[key])

    def make_dataloader(self, input_paths: Dict[str, Path]) -> DataLoader:
        return DataLoader(
            dpl.common.ImageFolderDataset(input_paths["crops"], normalize=True),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class EmocaFromImagesNode(EmocaNode):
    input_keys = ["images", "bboxes"]
    output_keys = ["shape", "tex", "exp", "pose", "cam", "light"]

    def make_dataloader(self, input_paths: Dict[str, Path]) -> DataLoader:
        return DataLoader(
            dpl.emoca.EmocaDataset(input_paths["images"], input_paths["bboxes"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
