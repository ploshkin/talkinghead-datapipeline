from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import skimage.io as io

from dpl import common
from dpl.processor.nodes.base import BaseNode
from dpl.processor.datatype import DataType

# To use JPEG compression in HDF5 you should install jpegHDF5 plugin.
# See: https://github.com/CARS-UChicago/jpegHDF5
HDF5_JPEG_PLUGIN = 32019


class H5BaseNode(BaseNode):

    def __init__(self, jpeg_quality: int = 95, recompute: bool = False) -> None:
        super().__init__(recompute)
        if not self.check_jpeg_plugin_exists():
            url = "https://github.com/CARS-UChicago/jpegHDF5"
            raise RuntimeError(f"jpegHDF5 plugin not found. Install it from {url!r}")

        self.quality = jpeg_quality

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
        for dt in self.input_types:
            if dt.key in input_paths:
                if dt.is_sequential():
                    paths = common.listdir(input_paths[dt.key], ext=dt.extensions())
                    images = np.stack([io.imread(path) for path in paths])
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

        fd.create_dataset(
            key,
            data=array,
            chunks=(1, *array.shape[1:]),
            compression=HDF5_JPEG_PLUGIN,
            compression_opts=(self.quality, width, height, rgb_flag),
        )

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
        DataType.VIDEO,
        DataType.IMAGES,
        DataType.BBOXES,
        DataType.CROPS,
        DataType.SHAPE,
        DataType.POSE,
        DataType.CAM,
        DataType.LIGHT,
        DataType.LANDMARKS3D,
    ]
    output_types = [DataType.SRC_SEQ]

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        path = self.get_output_path(output_paths)
        path.parent.mkdir(parents=True, exist_ok=True)

        fps = common.get_fps(input_paths["video"])
        with h5py.File(path, "w") as h5_file:
            h5_file.attrs.create("fps", fps)
            paths = self.filter_paths(input_paths, exclude_types=[DataType.VIDEO])
            self.write_data(h5_file, paths)

    def get_output_path(self, output_paths: Dict[str, Path]) -> Path:
        return output_paths["src_seq"]

    def filter_paths(
        self,
        paths: Dict[str, Path],
        *,
        include_types: Optional[List[DataType]] = None,
        exclude_types: Optional[List[DataType]] = None,
    ) -> Dict[str, Path]:
        if include_types is None and exclude_types is None:
            return paths

        if include_types is None:
            exclude_keys = set(dt.key for dt in exclude_types)
            return {
                key: path
                for key, path in paths.items()
                if key not in exclude_keys
            }

        if exclude_types is None:
            include_keys = set(dt.key for dt in include_types)
            return {
                key: path
                for key, path in paths.items()
                if key in include_keys
            }

        raise RuntimeError("Both `include_types` and `exclude_types` were specified.")
