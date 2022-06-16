from numbers import Number

import numpy as np


def to_integer(
    bbox: np.ndarray,
    preserve_size: bool = True,
    dtype: np.dtype = np.int64,
) -> np.ndarray:
    *coords, conf = bbox
    conf = round(conf * 100)
    if preserve_size:
        x_left, y_top, x_right, y_bottom = coords
        width = round(x_right - x_left)
        height = round(y_bottom - y_top)
        x_left = round(x_left)
        x_right = x_left + width
        y_top = round(y_top)
        y_bottom = y_top + height
    else:
        x_left, y_top, x_right, y_bottom = np.rint(coords).astype(dtype)
    return np.array([x_left, y_top, x_right, y_bottom, conf], dtype)


def to_square(bbox: np.ndarray) -> np.ndarray:
    x_left, y_top, x_right, y_bottom, conf = bbox
    width = x_right - x_left
    height = y_bottom - y_top
    size = max(width, height)

    dx = (size - width) / 2
    dy = (size - height) / 2

    return np.array(
        [x_left - dx, y_top - dy, x_right + dx, y_bottom + dy, conf],
        bbox.dtype,
    )


def pad_bbox(bbox: np.ndarray, pad: Number = 0) -> np.ndarray:
    x_left, y_top, x_right, y_bottom, conf = bbox
    if isinstance(pad, float):
        size = max(x_right - x_left, y_bottom - y_top)
        pad = round(size * pad)
    return np.array(
        [x_left - pad, y_top - pad, x_right + pad, y_bottom + pad, conf],
        bbox.dtype,
    )
