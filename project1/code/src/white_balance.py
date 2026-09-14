import numpy as np


def gray_world(inpu):
    dtype = np.result_type(inpu.dtype, np.float32)
    image = inpu.astype(dtype, copy=False)
    channel_means = np.mean(image, axis=(0, 1), dtype=np.float64)
    gray_mean = np.mean(channel_means)

    gains = np.ones_like(channel_means)
    np.divide(gray_mean, channel_means, out=gains, where=channel_means > 0)

    return image * gains.astype(dtype)
