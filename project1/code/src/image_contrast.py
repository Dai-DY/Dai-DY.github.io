import numpy as np


def automatic_contrast(inpu):
    dtype = np.result_type(inpu.dtype, np.float32)
    image = inpu.astype(dtype, copy=False)
    darkest = np.min(image)
    brightest = np.max(image)

    if brightest == darkest:
        return np.zeros_like(image)

    return (image - darkest) / (brightest - darkest)
