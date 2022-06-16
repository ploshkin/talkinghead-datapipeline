from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dpl.processor.nodes.base import BaseNode, BaseResource
import dpl.flame
import dpl.common


class FlameResource(BaseResource):
    def __init__(
        self,
        flame_model_path: Path,
        flame_lmk_embedding_path: Path,
        n_shape: int,
        n_exp: int,
        device: str,
    ) -> None:
        self.flame_model_path = flame_model_path
        self.flame_lmk_embedding_path = flame_lmk_embedding_path
        self.n_shape = n_shape
        self.n_exp = n_exp
        self.device = torch.device(device)
        self.reset()

    def __enter__(self) -> 'FlameResource':
        model = dpl.flame.FLAME(
            self.flame_model_path,
            self.flame_lmk_embedding_path,
            self.n_shape,
            self.n_exp,
        )
        self.model = model.to(self.device)
        self.model.eval()
        return self

    def reset(self) -> None:
        if hasattr(self, "model"):
            del self.model
        self.model = None


class FlameNode(BaseNode):
    input_keys = ["shape", "exp", "pose"]
    output_keys = ["verts"]

    def __init__(
        self,
        flame_model_path: Path,
        flame_lmk_embedding_path: Path,
        device: str,
        n_shape: int = 100,
        n_exp: int = 50,
        batch_size: int = 40,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.resource = FlameResource(
            flame_model_path, flame_lmk_embedding_path, n_shape, n_exp, device,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run_single(
        self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    ) -> None:
        verts = self.decode_flame(input_paths)
        output_paths["verts"].parent.mkdir(parents=True, exist_ok=True)
        np.save(output_paths["verts"], verts)

    def decode_flame(self, input_paths: Dict[str, Path]) -> np.ndarray:
        batched_verts = []
        dataloader = self.make_dataloader(input_paths)
        for index, batch in enumerate(dataloader):
            verts, _, _ = self.resource.model(
                shape_params=batch["shape"].to(self.resource.device),
                expression_params=batch["exp"].to(self.resource.device),
                pose_params=batch["pose"].to(self.resource.device),
            )
            batched_verts.append(verts.detach().cpu().numpy())
        return np.concatenate(batched_verts)

    def make_dataloader(self, input_paths: Dict[str, Path]) -> DataLoader:
        return DataLoader(
            dpl.common.NumpyDataset(input_paths),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
