from pathlib import Path
from typing import Dict, List

import face_alignment
import numpy as np
from torch.utils.data import DataLoader

from dpl.processor.nodes.base import BaseNode, BaseResource
import dpl.common


def nan_array(*shape: int) -> np.ndarray:
    return np.full(shape, np.nan)


def get_bbox_score(bbox: np.ndarray) -> float:
    x_left, y_top, x_right, y_bottom, conf = bbox
    return conf * (x_right - x_left) * (y_bottom - y_top)


def get_bbox(bboxes: List[np.ndarray]) -> np.ndarray:
    if not bboxes:
        return nan_array(5)
    return max(bboxes, key=get_bbox_score)


class FaceAlignmentResource(BaseResource):
    def __init__(self, filter_threshold: float, device: str) -> None:
        self.filter_threshold = filter_threshold
        self.device = device
        self.reset()

    def __enter__(self) -> 'FaceAlignmentResource':
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=self.device,
            face_detector_kwargs={
                "filter_threshold": self.filter_threshold,
            },
        )
        return self

    def reset(self) -> None:
        if hasattr(self, "fa"):
            del self.fa
        self.fa = None


class FaceDetectionNode(BaseNode):
    input_keys = ["images"]
    output_keys = ["raw_bboxes"]

    def __init__(
        self,
        filter_threshold: float,
        batch_size: int,
        num_workers: int,
        device: str,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resource = FaceAlignmentResource(filter_threshold, device)

    def run_single(
        self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    ) -> None:
        bboxes = self.detect_faces(input_paths["images"])
        self.save_outputs({"raw_bboxes": bboxes}, output_paths)

    def detect_faces(self, images_dir: Path) -> np.ndarray:
        dataloader = self.make_dataloader(images_dir)
        if len(dataloader) == 0:
            raise RuntimError(f"Empty directory: {str(images_dir)!r}")

        bboxes = []
        for images in dataloader:
            bboxes_batch = self.resource.fa.face_detector.detect_from_batch(
                images.to(self.resource.device),
            )
            bboxes.extend(map(get_bbox, bboxes_batch))

        return np.stack(bboxes)

    def save_outputs(self, outputs: Dict[str, np.ndarray], paths: Dict[str, Path]) -> None:
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
    input_keys = ["images"]
    output_keys = ["landmarks", "raw_bboxes"]

    def __init__(self, batch_size: int, num_workers: int, device: str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resource = FaceAlignmentResource(device)

    def run_single(
        self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    ) -> None:
        bboxes = self.detect_faces(input_paths["images"])

        landmarks = np.empty((len(bboxes), 68, 2))
        image_paths = dpl.common.listdir(input_paths["images"])
        for index, (bbox, path) in enumerate(zip(bboxes, image_paths)):
            if np.any(np.isnan(bbox)):
                landmarks[index] = nan_array(68, 2)
            else:
                lmks = self.resource.fa.get_landmarks_from_image(
                    path, detected_faces=[bbox],
                )
                landmarks[index] = lmks[0]

        outputs = {"landmarks": landmarks, "raw_bboxes": bboxes}
        self.save_outputs(outputs, output_paths)
