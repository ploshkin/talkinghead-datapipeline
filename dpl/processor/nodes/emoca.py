import collections
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal
import torch
from torch.utils.data import DataLoader

import dpl.emoca
import dpl.common
from dpl.processor.nodes.base import BaseNode, BaseResource
from dpl.processor.datatype import DataType


class EmocaResource(BaseResource):
    def __init__(self, weights_path: Path, device: str) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.device = torch.device(device)

    def load(self) -> None:
        self.model = dpl.emoca.EmocaInferenceWrapper().to(self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        super().load()

    def unload(self) -> None:
        del self.model
        super().unload()


class EmocaNode(BaseNode):
    input_types = [DataType.CROPS]
    output_types = [
        DataType.SHAPE,
        DataType.TEX,
        DataType.EXP,
        DataType.POSE,
        DataType.CAM,
        DataType.LIGHT,
    ]

    def __init__(
        self,
        weights_path: Path,
        device: str,
        use_smoothing: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        recompute: bool = False,
    ) -> None:
        super().__init__(recompute)
        self.resource = EmocaResource(weights_path, device)
        self.use_smoothing = use_smoothing
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        flame_codes = self.estimate_flame_codes(input_paths)
        if self.use_smoothing:
            flame_codes = self.smooth_flame_codes(flame_codes)
        self.save_outputs(flame_codes, output_paths)

    def estimate_flame_codes(
        self, input_paths: Dict[str, Path]
    ) -> Dict[str, np.ndarray]:
        keys = list(self.outputs.keys())

        batched_codes = collections.defaultdict(list)
        dataloader = self.make_dataloader(input_paths)
        for index, batch in enumerate(dataloader):
            codes = self.resource.model.encode(batch.to(self.resource.device))
            codes = {
                key: tensor.detach().cpu().numpy() for key, tensor in codes.items()
            }
            for key in keys:
                batched_codes[key].append(codes[key])

        return {key: np.concatenate(batched_codes[key]) for key in keys}

    def smooth_flame_codes(
        self, flame_codes: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        return {
            "shape": signal.savgol_filter(flame_codes["shape"], 25, 3, axis=0),
            "exp": flame_codes["exp"],
            "pose": signal.savgol_filter(flame_codes["pose"], 7, 3, axis=0),
            "cam": signal.savgol_filter(flame_codes["cam"], 11, 3, axis=0),
            "light": flame_codes["light"],
            "tex": flame_codes["tex"],
        }

    def save_outputs(
        self, outputs: Dict[str, np.ndarray], paths: Dict[str, Path]
    ) -> None:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, outputs[key])

    def make_dataloader(self, input_paths: Dict[str, Path]) -> DataLoader:
        return DataLoader(
            dpl.common.ImageFolderDataset(
                input_paths["crops"], size_hw=(224, 224), normalize=True
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class EmocaFromImagesNode(EmocaNode):
    input_types = [DataType.IMAGES, DataType.BBOXES]
    # output_types the same as in EmocaNode

    def make_dataloader(self, input_paths: Dict[str, Path]) -> DataLoader:
        return DataLoader(
            dpl.emoca.EmocaDataset(input_paths["images"], input_paths["bboxes"]),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
