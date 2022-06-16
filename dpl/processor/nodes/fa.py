from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import face_alignment
from joblib import Parallel, delayed
import numpy as np
from torch.utils.data import DataLoader

from dpl.processor.nodes.base import BaseNode, BaseResource
import dpl.fa

Pred = Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]


def get_bbox_index(bboxes: List[np.ndarray]) -> np.ndarray:
    "Get index of the bbox with max confidence."
    return np.argmax(np.array(bboxes)[:, 4])


def get_bbox(bboxes: List[np.ndarray]) -> np.ndarray:
    index = get_bbox_index(bboxes)
    return bboxes[index]


def nan_array(*shape: int) -> np.ndarray:
    return np.full(shape, np.nan)


# def recognize(
#     fa: face_alignment.FaceAlignment, path: Path, **kwargs: Any,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     return fa.get_landmarks_from_image(path, **kwargs)


class FaceAlignmentResource(BaseResource):
    def __init__(self, device: str) -> None:
        self.device = device
        self.reset()

    def __enter__(self) -> 'FaceAlignmentResource':
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=self.device,
            face_detector_kwargs={"filter_threshold": 0.9},
        )
        return self

    def reset(self) -> None:
        if hasattr(self, "fa"):
            del self.fa
        self.fa = None


class FaceAlignmentNode(BaseNode):
    input_keys = ["images"]
    output_keys = ["landmarks", "bboxes"]

    def __init__(self, batch_size: int, num_workers: int, device: str) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resource = FaceAlignmentResource(device)

    def run_single(
        self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    ) -> None:
        # TODO: use 'get_landmarks_from_batch'
        self._run_single_batched(input_paths, output_paths)

    def _run_single_batched(
        self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    ) -> None:
        dataloader = self._make_dataloader(input_paths["images"])
        if len(dataloader) == 0:
            return

        landmarks = []
        bboxes = []
        for index, images in enumerate(dataloader):
            batch_size = len(images)
            lmks, _, bbs = self.resource.fa.get_landmarks_from_batch(
                images.to(self.resource.device), return_bboxes=True,
            )
            landmarks.extend(
                [
                    lmks[i][: 68] if len(lmks[i]) > 0 else nan_array(68, 2)
                    for i in range(batch_size)
                ]
            )
            bboxes.extend(
                [
                    get_bbox(bbs[i]) if len(bbs[i]) > 0 else nan_array(5)
                    for i in range(batch_size)
                ]
            )
        outputs = {
            "landmarks": np.stack(landmarks),
            "bboxes": np.stack(bboxes),
        }
        self._save_outputs(outputs, output_paths)

    # def _run_single_imagewise(
    #     self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    # ) -> None:
    #     preds = self.resource.fa.get_landmarks_from_directory(
    #         str(input_paths["images"]),
    #         return_bboxes=True,
    #         show_progress_bar=False,
    #         recursive=False,
    #     )
    #     keys = sorted(preds.keys())
    #     outputs = {
    #         "landmarks": np.stack([self._get_lm(preds[key]) for key in keys]),
    #         "bboxes": np.stack([self._get_bbox(preds[key]) for key in keys]),
    #     }
    #     self._save_outputs(outputs, output_paths)

    # def _run_single_dumbparallel(
    #     self, input_paths: Dict[str, Path], output_paths: Dict[str, Path],
    # ) -> None:
    #     paths = sorted(map(str, input_paths["images"].iterdir()))
    #     with Parallel(n_jobs=4, prefer="threads") as parallel:
    #         predictions = parallel(
    #             delayed(recognize)(self.resource.fa, path, return_bboxes=True)
    #             for path in paths
    #         )
    #     outputs = {
    #         "landmarks": np.stack([self._get_lm(preds) for preds in predictions]),
    #         "bboxes": np.stack([self._get_bbox(preds) for preds in predictions]),
    #     }
    #     self._save_outputs(outputs, output_paths)

    def _save_outputs(self, outputs: Dict[str, np.ndarray], paths: Dict[str, Path]) -> None:
        for key, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, outputs[key])

#     def _get_lm(self, preds: Pred) -> np.ndarray:
#         if preds is None:
#             return nan_array(68, 2)
#         landmarks, _, bboxes = preds
#         index = get_bbox_index(bboxes)
#         return landmarks[index]

#     def _get_bbox(self, preds: Pred) -> np.ndarray:
#         if preds is None:
#             return nan_array(5)
#         _, _, bboxes = preds
#         index = get_bbox_index(bboxes)
#         return bboxes[index]

    def _make_dataloader(self, images: Path) -> DataLoader:
        return DataLoader(
            dpl.fa.FaceAlignmentDataset(images),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
