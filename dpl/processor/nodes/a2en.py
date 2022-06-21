from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from dpl import common
from dpl.processor.nodes.base import BaseNode
from dpl.processor.datatype import DataType
from dpl.wav2vec import utils as util


class A2enDatasetNode(BaseNode):
    input_types = [
        DataType.VIDEO,
        DataType.WAV2VEC,
        DataType.VOLUME,
        DataType.EXP,
        DataType.POSE,
        DataType.LANDMARKS3D,
    ]
    output_types = [DataType.A2EN]

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        data = {key: np.load(input_paths[key]) for key in ["wav2vec", "volume", "exp"]}
        data["jaw"] = np.load(input_paths["pose"])[:, 3]

        landmarks3d = np.load(input_paths["landmarks3d"])
        data.update(self.get_blinks_data(landmarks3d))

        fps = common.get_fps(input_paths["video"])
        num = len(data["exp"])

        if not data["volume"].size or not data["wav2vec"].size:
            raise RuntimeError("Audio is empty")

        data["volume"] = util.resample(data["volume"], num=num, source_fps=fps)
        data["wav2vec"] = util.resample(data["wav2vec"], num=num, source_fps=fps)

        output_paths["a2en"].parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_paths["a2en"], **data)

    def get_blinks_data(self, landmarks3d: np.ndarray) -> Dict[str, np.ndarray]:
        def l2_lmk(i: int, j: int) -> np.ndarray:
            return np.linalg.norm(landmarks3d[:, i] - landmarks3d[:, j], axis=1)

        return {
            "left_blink": (l2_lmk(37, 41) + l2_lmk(38, 40)) / (l2_lmk(36, 39) * 2),
            "right_blink": (l2_lmk(43, 47) + l2_lmk(44, 46)) / (l2_lmk(42, 45) * 2),
        }
