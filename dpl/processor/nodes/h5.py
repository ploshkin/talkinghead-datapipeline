from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import skimage.io as io

from dpl import common
from dpl.processor.nodes.base import BaseNode
from dpl.processor.datatype import DataType
from dpl.processor import utils
from dpl.wav2vec import utils as wav2vec_utils

# To use JPEG compression in HDF5 you should install jpegHDF5 plugin.
# See: https://github.com/CARS-UChicago/jpegHDF5
HDF5_JPEG_PLUGIN = 32019


class H5BaseNode(BaseNode):
    def __init__(
        self,
        jpeg_quality: int = 95,
        batch_size: Optional[int] = None,
        recompute: bool = False,
    ) -> None:
        super().__init__(recompute)
        if not self.check_jpeg_plugin_exists():
            url = "https://github.com/CARS-UChicago/jpegHDF5"
            raise RuntimeError(f"jpegHDF5 plugin not found. Install it from {url!r}")

        self.quality = jpeg_quality

        self.batch_size = batch_size
        if self.batch_size is not None:
            self.batch_size = max(1, self.batch_size)

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        path = self.get_output_path(output_paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5_file:
            self.write_data(h5_file, input_paths)

    def write_data(self, fd: h5py.File, input_paths: Dict[str, Path]) -> None:
        for dt in filter(lambda dt: dt.key in input_paths, self.input_types):
            if dt.is_sequential():
                paths = common.listdir(input_paths[dt.key], ext=dt.extensions())
                if self.batch_size is None:
                    images = np.stack([io.imread(path) for path in paths])
                    self.add_images(fd, dt.key, images)
                else:
                    for start in range(0, len(paths), self.batch_size):
                        slc = slice(start, start + self.batch_size)
                        images = np.stack([io.imread(path) for path in paths[slc]])
                        self.add_images(fd, dt.key, images)
            else:
                array = np.load(input_paths[dt.key])
                fd.create_dataset(dt.key, data=array, compression="gzip")

    def add_images(self, fd: h5py.File, key: str, array: np.ndarray) -> None:
        is_color = len(array.shape) == 4 and array.shape[3] == 3
        if not is_color and len(array.shape) == 4:
            array = array[..., 0]

        rgb_flag = 1 if is_color else 0
        height, width = array.shape[1:3]

        if key not in fd:
            image_shape = (height, width, 3) if is_color else (height, width)
            fd.create_dataset(
                key,
                data=array,
                chunks=(1, *array.shape[1:]),
                compression=HDF5_JPEG_PLUGIN,
                compression_opts=(self.quality, width, height, rgb_flag),
                maxshape=(None, *image_shape),
            )
        else:
            fd[key].resize(len(fd[key]) + len(array), axis=0)
            fd[key][-len(array) :] = array

    def check_jpeg_plugin_exists(self) -> bool:
        # `/usr/local/hdf5/lib/plugin` is the default HDF5 plugin directory.
        # See: https://support.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters
        default_path = Path("/usr/local/hdf5/lib/plugin/libjpeg_h5plugin.so")
        return default_path.exists()

    def get_output_path(self, output_paths: Dict[str, Path]) -> Path:
        raise NotImplementedError


class Vid2vidDatasetNode(H5BaseNode):
    input_types = [
        DataType.CROPS,
        DataType.RENDER_UV,
        DataType.RENDER_NORMAL,
    ]
    output_types = [DataType.VID2VID]

    def get_output_path(self, output_paths: Dict[str, Path]) -> Path:
        return output_paths["vid2vid"]


class SourceSequenceNode(H5BaseNode):
    input_types = [
        DataType.IMAGES,
        DataType.BBOXES,
        DataType.CROPS,
        DataType.SHAPE,
        DataType.POSE,
        DataType.CAM,
        DataType.LIGHT,
        DataType.LANDMARKS3D,
        DataType.RENDER_UV,
        DataType.RENDER_NORMAL,
    ]
    output_types = [DataType.SRC_SEQ]

    def __init__(
        self,
        fps: float = 30.0,
        jpeg_quality: int = 95,
        batch_size: Optional[int] = None,
        recompute: bool = False,
    ) -> None:
        super().__init__(jpeg_quality, batch_size, recompute)
        self.fps = fps

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        path = self.get_output_path(output_paths)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as h5_file:
            h5_file.attrs.create("fps", self.fps)
            self.write_data(h5_file, input_paths)

    def get_output_path(self, output_paths: Dict[str, Path]) -> Path:
        return output_paths["src_seq"]


class Vid2vidAudioNode(H5BaseNode):
    input_types = [
        DataType.LANDMARKS,
        DataType.WAV2VEC,
        DataType.VOLUME,
        DataType.CROPS,
        DataType.RENDER_UV,
        DataType.RENDER_NORMAL,
    ]
    output_types = [DataType.VID2VID_AUDIO]

    def __init__(
        self,
        fps: float,
        window_size: int = 16,
        jpeg_quality: int = 95,
        batch_size: Optional[int] = None,
        recompute: bool = False,
    ) -> None:
        super().__init__(jpeg_quality, batch_size, recompute)
        self.fps = fps
        self.window_size = window_size

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        path = self.get_output_path(output_paths)
        path.parent.mkdir(parents=True, exist_ok=True)

        num = len(common.listdir(input_paths["crops"]))

        volume = wav2vec_utils.resample(np.load(input_paths["volume"]), num, self.fps)
        volume = self.average_features(volume, self.window_size)

        wav2vec = wav2vec_utils.resample(np.load(input_paths["wav2vec"]), num, self.fps)
        wav2vec = self.average_features(wav2vec, self.window_size)

        landmarks = np.load(input_paths["landmarks"])
        blinks = utils.get_blinks_data(landmarks)

        concat_features = np.concatenate(
            [
                wav2vec,
                volume[:, np.newaxis],
                blinks["left_blink"][:, np.newaxis],
                blinks["right_blink"][:, np.newaxis],
            ],
            axis=1,
        )

        with h5py.File(path, "w") as h5_file:
            data_paths = {
                key: path
                for key, path in input_paths.items()
                if key in ["crops", "render_uv", "render_normal"]
            }
            self.write_data(h5_file, data_paths)

            h5_file.create_dataset("wav2vec", data=wav2vec, compression="gzip")
            h5_file.create_dataset("volume", data=volume, compression="gzip")
            h5_file.create_dataset(
                "average_blink", data=blinks["average_blink"], compression="gzip"
            )
            h5_file.create_dataset(
                "audio_blink_feature", data=concat_features, compression="gzip"
            )

    def get_output_path(self, output_paths: Dict[str, Path]) -> Path:
        return output_paths["vid2vid_audio"]

    @staticmethod
    def average_features(features: np.ndarray, window_size: int) -> np.ndarray:
        return utils.as_windowed(features, window_size, mode="edge").mean(axis=1)
