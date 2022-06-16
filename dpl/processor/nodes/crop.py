from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from joblib import Parallel, delayed
import numpy as np
from PIL import Image
from tqdm import tqdm

import dpl.common
from dpl.processor.nodes.base import BaseNode, BaseResource


class CropResize:
    def __init__(
        self,
        size_hw: Tuple[int, int],
        resample: Image.Resampling = Image.Resampling.LANCZOS,
        save_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.size_hw = size_hw
        self.resample = resample
        self.save_kwargs = save_kwargs or {}

    def __call__(self, source: Path, target: Path, bbox: np.ndarray) -> None:
        Image.open(source).crop(bbox[:4]).resize(
            self.size_hw,
            self.resample,
        ).save(target, **self.save_kwargs)


class CropNode(BaseNode):
    input_keys = ["images", "bboxes"]
    output_keys = ["crops"]

    def __init__(
        self,
        size_hw: Tuple[int, int],
        num_jobs: int = 32,
        save_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cropper = CropResize(size_hw, save_kwargs=save_kwargs)
        self.num_jobs = num_jobs

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        images_dir = input_paths["images"]

        bboxes = np.load(input_paths["bboxes"])
        if np.any(np.isnan(bboxes)):
            raise RuntimeError(
                f"NaN values in bboxes, source = '{input_paths['bboxes']}'."
            )

        output_paths["crops"].mkdir(parents=True, exist_ok=True)
        image_paths = dpl.common.listdir(images_dir)
        crop_paths = [
            output_paths["crops"] / path.relative_to(images_dir) for path in image_paths
        ]

        iterator = zip(image_paths, crop_paths, bboxes)
        with Parallel(n_jobs=self.num_jobs, prefer="processes") as parallel:
            parallel(delayed(self.cropper)(*args) for args in iterator)
