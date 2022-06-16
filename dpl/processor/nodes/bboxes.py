from pathlib import Path
from typing import Dict

import numpy as np

from dpl.processor.nodes.base import BaseNode
import dpl.processor.utils as util


class TransformRawBboxesNode(BaseNode):
    input_keys = ["raw_bboxes"]
    output_keys = ["bboxes"]

    def __init__(self, extend_factor: float) -> None:
        super().__init__()
        self.pad = extend_factor

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        raw_bboxes = np.load(input_paths["raw_bboxes"])
        if np.any(np.isnan(raw_bboxes)):
            raise RuntimeError(
                f"NaN values in bboxes, source = '{input_paths['raw_bboxes']}'."
            )

        bboxes = np.stack([self._transform_bbox(bbox) for bbox in raw_bboxes])

        output_paths["bboxes"].parent.mkdir(parents=True, exist_ok=True)
        np.save(output_paths["bboxes"], bboxes)

    def _transform_bbox(
        self, bbox: np.ndarray, dtype: np.dtype = np.int64
    ) -> np.ndarray:
        return util.to_integer(
            util.pad_bbox(util.to_square(bbox), self.pad),
            dtype=dtype,
        )
