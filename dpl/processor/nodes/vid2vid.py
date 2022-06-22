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


class Vid2vidDatasetNode(BaseNode):
    input_types = [
        DataType.CROPS,
        DataType.RENDER_UV,
        DataType.RENDER_NORMAL,
    ]
    output_types = [DataType.VID2VID]

    def __init__(self, quality: int = 95, recompute: bool = False) -> None:
        super().__init__(recompute)
        if not self._check_jpeg_plugin_exists():
            url = "https://github.com/CARS-UChicago/jpegHDF5"
            raise RuntimeError(f"jpegHDF5 plugin not found. Install it from {url!r}")

        self.quality = quality

    def run_single(
        self,
        input_paths: Dict[str, Path],
        output_paths: Dict[str, Path],
    ) -> None:
        output_paths["vid2vid"].parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_paths["vid2vid"], "w") as h5_file:
            for dt in self.input_types:
                if dt.is_sequential():
                    paths = common.listdir(input_paths[dt.key], ext=dt.extensions())
                    images = np.stack([io.imread(path) for path in paths])
                    self._add_images(h5_file, dt.key, images)
                else:
                    array = np.load(input_paths[dt.key])
                    h5_file.create_dataset(key, data=array, compression="gzip")

    def _add_images(self, fd: h5py.File, key: str, array: np.ndarray) -> None:
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

    def _check_jpeg_plugin_exists(self) -> bool:
        # `/usr/local/hdf5/lib/plugin` is the default HDF5 plugin directory.
        # See: https://support.hdfgroup.org/HDF5/doc/Advanced/DynamicallyLoadedFilters
        default_path = Path("/usr/local/hdf5/lib/plugin/libjpeg_h5plugin.so")
        return default_path.exists()
