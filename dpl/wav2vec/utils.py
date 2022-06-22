import numpy as np
import scipy.interpolate as interpolate


def resample(y: np.ndarray, num: int, source_fps: float) -> np.ndarray:
    length_sec = len(y) / source_fps
    target_fps = num / length_sec

    dx = 1 / (source_fps * 2)
    dx_hat = 1 / (target_fps * 2)

    x = np.linspace(dx, length_sec, len(y))
    x_hat = np.linspace(dx_hat, length_sec, num)

    interp_fn = interpolate.interp1d(
        x, y, axis=0, assume_sorted=True, fill_value="extrapolate"
    )
    return interp_fn(x_hat)
