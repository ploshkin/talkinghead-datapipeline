from pathlib import Path
from typing import Dict, List, Optional

import face_alignment
import numpy as np
from torch.utils.data import DataLoader

from dpl.processor.nodes.base import BaseNode, BaseResource
from dpl.processor.datatype import DataType
import dpl.common


def nan_array(*shape: int) -> np.ndarray:
    return np.full(shape, np.nan)


def get_bbox_score(bbox: np.ndarray) -> float:
    return bbox[4]


def get_bbox(bboxes: List[np.ndarray]) -> np.ndarray:
    if not bboxes:
        return nan_array(5)
    return max(bboxes, key=get_bbox_score)


class FaceAlignmentResource(BaseResource):
    def __init__(self, device: str, filter_threshold: float = 0.5) -> None:
        super().__init__()
        self.filter_threshold = filter_threshold
        self.device = device

    def load(self) -> None:
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=self.device,
            face_detector_kwargs={
                "filter_threshold": self.filter_threshold,
            },
        )
        super().load()

    def unload(self) -> None:
        del self.fa
        super().unload()


class FaceDetectionNode(BaseNode):
    input_types = [DataType.IMAGES]
    output_types = [DataType.RAW_BBOXES]

    def __init__(
        self,
        filter_threshold: float,
        batch_size: int,
        num_workers: int,
        device: str,
        recompute: bool = False,
    ) -> None:
        super().__init__(recompute)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resource = FaceAlignmentResource(device, filter_threshold)

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        bboxes = self.detect_faces(input_paths["images"])
        self.save_outputs({"raw_bboxes": bboxes}, output_paths)

    def detect_faces(self, images_dir: Path) -> np.ndarray:
        dataloader = self.make_dataloader(images_dir)
        if len(dataloader) == 0:
            raise RuntimeError(f"Empty directory: {str(images_dir)!r}")

        bboxes = []
        for images in dataloader:
            bboxes_batch = self.resource.fa.face_detector.detect_from_batch(images)
            bboxes.extend(map(get_bbox, bboxes_batch))

        return np.stack(bboxes)

    def save_outputs(
        self, outputs: Dict[str, np.ndarray], paths: Dict[str, Path]
    ) -> None:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, outputs[key])

    def make_dataloader(self, images_dir: Path) -> DataLoader:
        return DataLoader(
            dpl.common.ImageFolderDataset(images_dir),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


class FaceAlignmentNode(FaceDetectionNode):
    # input_types the same as in FaceDetectionNode.
    output_types = [DataType.LANDMARKS, DataType.RAW_BBOXES]

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        bboxes = self.detect_faces(input_paths["images"])

        landmarks = np.empty((len(bboxes), 68, 2))
        image_paths = dpl.common.listdir(input_paths["images"], ext=[".jpg"])
        for index, (bbox, path) in enumerate(zip(bboxes, image_paths)):
            if np.any(np.isnan(bbox)):
                landmarks[index] = nan_array(68, 2)
            else:
                lmks = self.resource.fa.get_landmarks_from_image(
                    str(path),
                    detected_faces=[bbox],
                )
                landmarks[index] = lmks[0]

        outputs = {"landmarks": landmarks, "raw_bboxes": bboxes}
        self.save_outputs(outputs, output_paths)


class FaceLandmarksNode(BaseNode):
    input_types = [DataType.IMAGES, DataType.RAW_BBOXES]
    output_types = [DataType.LANDMARKS]

    def __init__(self, device: str, recompute: bool = False) -> None:
        super().__init__(recompute)
        self.resource = FaceAlignmentResource(device)

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        bboxes = np.load(input_paths["raw_bboxes"])

        landmarks = np.empty((len(bboxes), 68, 2))
        image_paths = dpl.common.listdir(input_paths["images"], ext=[".jpg"])
        for index, (bbox, path) in enumerate(zip(bboxes, image_paths)):
            if np.any(np.isnan(bbox)):
                landmarks[index] = nan_array(68, 2)
            else:
                lmks = self.resource.fa.get_landmarks_from_image(
                    str(path),
                    detected_faces=[bbox],
                )
                landmarks[index] = lmks[0]

        self.save_outputs({"landmarks": landmarks}, output_paths)

    def save_outputs(
        self, outputs: Dict[str, np.ndarray], paths: Dict[str, Path]
    ) -> None:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, outputs[key])
