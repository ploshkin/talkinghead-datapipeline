import os
import tempfile
from typing import Tuple, Union

import h5py
import numpy as np

HDF5_JPEG_PLUGIN = 32019


BatchShape = Union[Tuple[int, int, int], Tuple[int, int, int, int]]


def deduce_shape(count: int, size_hw: Tuple[int, int], color_mode: str) -> BatchShape:
    if color_mode == "grayscale":
        return (count, *size_hw)
    elif color_mode == "rgb":
        return (count, *size_hw, 3)
    raise ValueError(f"Unknown color_mode: {color_mode!r}")


def generate_random_image_batch(
    count: int, size_hw: Tuple[int, int], color_mode: str = "rgb"
) -> np.ndarray:
    shape = deduce_shape(count, size_hw, color_mode)
    return np.random.randint(np.iinfo(np.uint8).max + 1, size=shape, dtype=np.uint8)


def generate_black_image_batch(
    count: int, size_hw: Tuple[int, int], color_mode: str = "rgb"
) -> np.ndarray:
    shape = deduce_shape(count, size_hw, color_mode)
    return np.zeros(shape, dtype=np.uint8)


def test_jpeghdf5_random() -> None:
    num_images = 100
    height, width = 320, 320
    mode = "rgb"
    images = generate_random_image_batch(num_images, (height, width), mode)
    quality = 80

    with tempfile.TemporaryDirectory() as tmpdir:
        path_no_comp = os.path.join(tmpdir, "no_compression.h5")
        with h5py.File(path_no_comp, "w") as ofile:
            ofile.create_dataset("images", data=images)
        size_no_comp = os.path.getsize(path_no_comp)

        path_lzf = os.path.join(tmpdir, "lzf.h5")
        with h5py.File(path_lzf, "w") as ofile:
            ofile.create_dataset("images", data=images, compression="lzf")
        size_lzf = os.path.getsize(path_lzf)

        path_jpeg = os.path.join(tmpdir, "jpeg.h5")
        with h5py.File(path_jpeg, "w") as ofile:
            ofile.create_dataset(
                "images",
                data=images,
                chunks=(1, *images.shape[1:]),
                compression=HDF5_JPEG_PLUGIN,
                compression_opts=(quality, width, height, int(mode == "rgb")),
            )
        size_jpeg = os.path.getsize(path_jpeg)

    assert size_jpeg * 4 < size_no_comp
    assert size_jpeg * 4 < size_lzf


def test_jpeghdf5_black() -> None:
    num_images = 100
    height, width = 320, 320
    mode = "rgb"
    images = generate_black_image_batch(num_images, (height, width), mode)
    quality = 80

    with tempfile.TemporaryDirectory() as tmpdir:
        path_no_comp = os.path.join(tmpdir, "no_compression.h5")
        with h5py.File(path_no_comp, "w") as ofile:
            ofile.create_dataset("images", data=images)
        size_no_comp = os.path.getsize(path_no_comp)

        path_lzf = os.path.join(tmpdir, "lzf.h5")
        with h5py.File(path_lzf, "w") as ofile:
            ofile.create_dataset("images", data=images, compression="lzf")
        size_lzf = os.path.getsize(path_lzf)

        path_jpeg = os.path.join(tmpdir, "jpeg.h5")
        with h5py.File(path_jpeg, "w") as ofile:
            ofile.create_dataset(
                "images",
                data=images,
                chunks=(1, *images.shape[1:]),
                compression=HDF5_JPEG_PLUGIN,
                compression_opts=(quality, width, height, int(mode == "rgb")),
            )
        size_jpeg = os.path.getsize(path_jpeg)

    assert size_lzf < size_no_comp
    assert size_jpeg * 10 < size_no_comp
    assert size_jpeg * 1.5 < size_lzf
