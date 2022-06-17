import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from joblib import Parallel, delayed
import numpy as np
import skimage.io as io
import torch
from torch.utils.data import DataLoader

from dpl.processor.nodes.base import BaseNode, BaseResource
from dpl.processor.datatype import DataType
from dpl.rendering import utils as util
import dpl.common


class RenderingResource(BaseResource):
    def __init__(
        self,
        head_template_path: Path,
        image_size: int,
        uv_size: int,
        batch_size: int,
        device: str,
    ) -> None:
        self.head_template_path = head_template_path
        self.image_size = image_size
        self.uv_size = uv_size
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.reset()

    def __enter__(self) -> "RenderingResource":
        model = dpl.rendering.SRenderY(
            image_size=self.image_size,
            obj_filename=self.head_template_path,
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.albedo = util.get_radial_uv(
            self.uv_size,
            self.batch_size,
            self.device,
        )
        return self

    def reset(self) -> None:
        if hasattr(self, "model"):
            del self.model
        self.model = None
        self.albedo = None


class RenderingNode(BaseNode):
    input_types = [DataType.VERTS, DataType.CAM, DataType.LIGHT]
    output_types = [DataType.RENDER_NORMAL, DataType.RENDER_UV]

    NAME_MAPPING = {
        DataType.RENDER_UV.key: "images",
        DataType.RENDER_NORMAL.key: "normal_images",
    }

    def __init__(
        self,
        head_template_path: Path,
        image_size: int,
        device: str,
        uv_size: int = 256,
        batch_size: int = 4,
        num_workers: int = 4,
        num_save_jobs: int = 8,
    ) -> None:
        super().__init__()

        self.resource = RenderingResource(
            head_template_path,
            image_size,
            uv_size,
            batch_size,
            device,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_jobs = num_save_jobs

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        renders = self.render(input_paths)
        self.save_renders(renders, output_paths)

    def render(self, input_paths: Dict[str, Path]) -> Dict[str, List[np.ndarray]]:
        render_lists = collections.defaultdict(list)
        dataloader = self.make_dataloader(input_paths)

        for batch_index, batch in enumerate(dataloader):
            batch_size = len(batch["verts"])
            trans_verts = util.batch_orth_proj(
                batch["verts"],
                batch["cam"],
            )
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
            renders = self.resource.model(
                batch["verts"].to(self.resource.device),
                trans_verts.to(self.resource.device),
                self.resource.albedo[:batch_size],
                batch["light"].to(self.resource.device),
            )
            for key in self.__class__.output_keys:
                name = RenderingNode.NAME_MAPPING[key]
                render_lists[key].extend(map(util.to_image, renders[name]))

        return render_lists

    def save_renders(
        self,
        renders: Dict[str, List[np.ndarray]],
        output_paths: Dict[str, Path],
    ) -> None:
        for key, output_dir in output_paths.items():
            output_dir.mkdir(parents=True, exist_ok=True)
            with Parallel(n_jobs=self.num_jobs) as parallel:
                parallel(
                    delayed(io.imsave)(
                        output_dir / f"{i:06d}.jpg",
                        renders[key][i],
                        plugin="pil",
                        quality=95,
                    )
                    for i in range(len(renders[key]))
                )

    def make_dataloader(self, input_paths: Dict[str, Path]) -> DataLoader:
        return DataLoader(
            dpl.common.NumpyDataset(input_paths),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
