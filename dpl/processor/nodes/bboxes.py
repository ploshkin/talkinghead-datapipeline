from pathlib import Path
from typing import Dict

import numpy as np
from scipy import signal

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


class EmocaLikeBboxesNode(BaseNode):
    input_types = [DataType.LANDMARKS]
    output_types = [DataType.BBOXES]

    def __init__(
        self,
        scale: float = 1.25,
        window_size: int = 5,
        recompute: bool = False
    ) -> None:
        super().__init__(recompute)
        self.scale = scale
        self.window_size = window_size

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
        bboxes = self.smooth_bboxes(self.get_bboxes(landmarks))
        output_paths["bboxes"].parent.mkdir(parents=True, exist_ok=True)
        np.save(output_paths["bboxes"], bboxes)

    def get_bboxes(self, landmarks: np.ndarray) -> np.ndarray:
        offset_left = self.window_size // 2
        offset_right = self.window_size - offset_left

        bboxes = np.zeros((len(landmarks), 4), dtype=np.int64)
        # TODO: Optimize this loop: O(N x window_size) -> O(N)
        for i, _ in enumerate(landmarks):
            slc = slice(max(0, i - offset_left), min(len(landmarks), i + offset_right))

            left = np.min(landmarks[slc, :, 0])
            right = np.max(landmarks[slc, :, 0])
            top = np.min(landmarks[slc, :, 1])
            bottom = np.max(landmarks[slc, :, 1])

            size = (right - left + bottom - top) / 2 * 1.1
            radius = int(self.scale * size / 2.0)

            xc = int((right + left) / 2.0)
            yc = int((bottom + top) / 2.0)

            bboxes[i] = [xc - radius, yc - radius, xc + radius, yc + radius]

        return bboxes

    def smooth_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        sizes_hor = bboxes[..., 2] - bboxes[..., 0]
        sizes_ver = bboxes[..., 3] - bboxes[..., 1]

        assert (sizes_hor == sizes_ver).all()

        sizes = sizes_hor.copy()
        radiuses = sizes / 2

        xcs = bboxes[..., 0] + radiuses
        ycs = bboxes[..., 1] + radiuses

        xcs = np.rint(signal.savgol_filter(xcs, 25, 3)).astype(np.int64)
        ycs = np.rint(signal.savgol_filter(ycs, 25, 3)).astype(np.int64)

        radiuses = np.ceil(signal.savgol_filter(radiuses, 15, 3)).astype(np.int64)

        return np.array(
            [
                [xc - radius, yc - radius, xc + radius, yc + radius]
                for xc, yc, radius in zip(xcs, ycs, radiuses)
            ],
            dtype=np.int64
        )
