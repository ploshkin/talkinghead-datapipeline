from pathlib import Path
from typing import Dict, List, Optional, Tuple

import face_alignment
import numpy as np

Pred = Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]


def get_bbox_index(bboxes: List[np.ndarray]) -> np.ndarray:
    "Get index of the bbox with max confidence."
    return np.argmax(np.array(bboxes)[:, 4])


def nan_array(*shape: int) -> np.ndarray:
    return np.full(shape, np.nan)


class FaceLandmarksEstimator:
    def __init__(
        self,
        name: Optional[str] = None,
        extensions: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        self.extensions = [".png", ".jpg"] or extensions
        self.name = name or "face_data"
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D,
            flip_input=False,
            device=device,
        )

    def compute_landmarks(
        self,
        video_paths: List[Path],
        output_paths: Optional[List[Path]] = None,
        concat_results: bool = False,
    ):
        if output_paths is None:
            suffix = ".npz" if concat_results else ""
            output_paths = self._make_output_paths(video_paths, suffix)

        if len(video_paths) != len(output_paths):
            ineq = f"{len(video_paths)} != {len(output_paths)}"
            message = f"Number of input and output paths should match, got {ineq}"
            raise RuntimeError(message)

        for video_path, output_path in zip(video_paths, output_paths):
            predictions = self._fa.get_landmarks_from_directory(
                str(video_path),
                return_bboxes=True,
                extensions=self.extensions,
                show_progress_bar=False,
                recursive=False,
            )
            self._save_predictions(predictions, output_path, concat_results)

    def _make_output_paths(self, paths: List[Path], suffix: str) -> List[Path]:
        return [path.with_name(self.name).with_suffix(suffix) for path in paths]

    def _save_predictions(
        self,
        predictions: Dict[str, Pred],
        path: Path,
        concat_results: bool,
    ) -> None:
        if concat_results:
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(predictions.keys())
            landmarks = np.stack([self._get_landmarks(predictions[key]) for key in keys])
            bboxes = np.stack([self._get_bbox(predictions[key]) for key in keys])
            np.savez(path, landmarks=landmarks, bboxes=bboxes)

        else:
            path.mkdir(parents=True, exist_ok=True)
            for key, preds in predictions.items():
                np.savez(
                    path / f"{Path(key).with_suffix('').name}.npz",
                    landmarks=self._get_landmarks(preds),
                    bboxes=self._get_bbox(preds),
                )

    def _get_landmarks(self, preds: Pred) -> np.ndarray:
        if preds is None:
            return nan_array(68, 2)
        landmarks, _, bboxes = preds
        index = get_bbox_index(bboxes)
        return landmarks[index]

    def _get_bbox(self, preds: Pred) -> np.ndarray:
        if preds is None:
            return nan_array(5)
        _, _, bboxes = preds
        index = get_bbox_index(bboxes)
        return bboxes[index]
