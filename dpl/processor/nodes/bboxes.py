from pathlib import Path
from typing import Dict

import numpy as np

from dpl.processor.nodes.base import BaseNode
from dpl.processor.datatype import DataType
import dpl.processor.utils as util


class TransformRawBboxesNode(BaseNode):
    input_types = [DataType.RAW_BBOXES]
    output_types = [DataType.BBOXES]

    def __init__(self, extend_factor: float, recompute: bool = False) -> None:
        super().__init__(recompute)
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


class FixedBboxesNode(BaseNode):
    input_types = [DataType.LANDMARKS]
    output_types = [DataType.BBOXES]

    def __init__(self, scale: float = 1.25, recompute: bool = False) -> None:
        super().__init__(recompute)
        self.scale = scale

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        landmarks = np.load(input_paths["landmarks"])
        if np.any(np.isnan(landmarks)):
            raise RuntimeError(
                f"NaN values in landmarks, source = '{input_paths['landmarks']}'."
            )
        bboxes = self.get_bboxes(landmarks)
        output_paths["bboxes"].parent.mkdir(parents=True, exist_ok=True)
        np.save(output_paths["bboxes"], bboxes)

    def get_bboxes(self, landmarks: np.ndarray) -> np.ndarray:
        left = np.min(landmarks[..., 0])
        right = np.max(landmarks[..., 0])
        top = np.min(landmarks[..., 1])
        bottom = np.max(landmarks[..., 1])

        size = (right - left + bottom - top) / 2 * 1.1
        radius = int(self.scale * size / 2.0)

        xc = int((right + left) / 2.0)
        yc = int((bottom + top) / 2.0)

        bboxes = np.zeros((len(landmarks), 4), dtype=np.int64)
        bboxes[:] = [xc - radius, yc - radius, xc + radius, yc + radius]
        return bboxes
