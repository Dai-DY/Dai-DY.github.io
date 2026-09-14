import numpy as np


PREWITT_X = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.float32)

PREWITT_Y = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1],
], dtype=np.float32)

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)

SOBEL_Y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
], dtype=np.float32)

ROBERTS_X = np.array([
    [0, 1],
    [-1, 0],
], dtype=np.float32)

ROBERTS_Y = np.array([
    [1, 0],
    [0, -1],
], dtype=np.float32)


GRADIENT_FILTERS = {
    "prewitt": (PREWITT_X, PREWITT_Y),
    "sobel": (SOBEL_X, SOBEL_Y),
    "roberts": (ROBERTS_X, ROBERTS_Y),
}


def filter_2d(inpu, kernel):
    height, width = inpu.shape[:2]
    kernel_height, kernel_width = kernel.shape
    pad_top = kernel_height // 2
    pad_bottom = kernel_height - pad_top - 1
    pad_left = kernel_width // 2
    pad_right = kernel_width - pad_left - 1

    pad_width = [(pad_top, pad_bottom), (pad_left, pad_right)]
    if inpu.ndim == 3:
        pad_width.append((0, 0))

    dtype = np.result_type(inpu.dtype, kernel.dtype, np.float32)
    padded = np.pad(
        inpu.astype(dtype, copy=False),
        pad_width,
        mode="edge",
    )
    response = np.zeros(inpu.shape, dtype=dtype)

    for row in range(kernel_height):
        for col in range(kernel_width):
            response += kernel[row, col] * padded[
                row:row + height,
                col:col + width,
                ...,
            ]

    return response


def image_gradient(inpu, method="sobel"):
    method = method.lower()
    if method not in GRADIENT_FILTERS:
        choices = ", ".join(GRADIENT_FILTERS)
        raise ValueError(f"Unknown gradient method '{method}'. Choose from: {choices}")

    kernel_x, kernel_y = GRADIENT_FILTERS[method]
    gradient_x = filter_2d(inpu, kernel_x)
    gradient_y = filter_2d(inpu, kernel_y)
    magnitude = np.hypot(gradient_x, gradient_y)

    return gradient_x, gradient_y, magnitude


def prewitt(inpu):
    return image_gradient(inpu, method="prewitt")


def sobel(inpu):
    return image_gradient(inpu, method="sobel")


def roberts(inpu):
    return image_gradient(inpu, method="roberts")
