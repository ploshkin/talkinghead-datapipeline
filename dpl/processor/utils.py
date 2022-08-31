from typing import Dict
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


def l2_batch(points: np.ndarray, first: int, second: int) -> np.ndarray:
    return np.linalg.norm(points[:, first] - points[:, second], axis=1)


def get_blinks_data(lmks: np.ndarray) -> Dict[str, np.ndarray]:
    l2_lmk = lambda first, second: l2_batch(lmks, first, second)
    left_blink = (l2_lmk(37, 41) + l2_lmk(38, 40)) / (l2_lmk(36, 39) * 2)
    right_blink = (l2_lmk(43, 47) + l2_lmk(44, 46)) / (l2_lmk(42, 45) * 2)
    return {
        "left_blink": left_blink,
        "right_blink": right_blink,
        "average_blink": (left_blink + right_blink) / 2,
    }


def as_windowed(x: np.ndarray, size: int, **kwargs) -> np.ndarray:
    """Creates windowed view to the input array: (N, *dims) -> (N, size, *dims).

    Parameters
    ----------
    x : np.ndarray of shape (N, *dims)
        Input array.

    size : int
        Window size.

    **kwargs : Any
        Keyword arguments for np.pad

    Returns
    -------
    result : np.ndarray of shape (N, size, *dims)
        Windowed input array.
    """
    n_items, *dims = x.shape
    pad_size = (size - size // 2, size // 2)
    zero_pads = [(0, 0)] * len(dims)
    padded = np.pad(x, [pad_size, *zero_pads], **kwargs)

    first_stride, *strides = padded.strides
    view = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_items, size, *dims),
        strides=(first_stride, first_stride, *strides),
    )
    return np.array(view)
