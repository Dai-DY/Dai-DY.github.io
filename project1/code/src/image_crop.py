import numpy as np

from src import image_gradient


def crop_common_overlap(inpu, offsets):
    height, width = inpu.shape[:2]
    left = 0
    right = width
    top = 0
    bottom = height

    for dx, dy in offsets:
        left = max(left, dx)
        right = min(right, width + dx)
        top = max(top, dy)
        bottom = min(bottom, height + dy)

    return inpu[top:bottom, left:right]


def smooth_profile(profile):
    window_size = max(3, int(profile.size * 0.005))
    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(profile, kernel, mode="same")


def find_start_border(
    gradient_profile,
    intensity_profile,
    search_size,
    band_size,
    dark_limit,
    minimum_contrast,
):
    best_score = -np.inf
    best_border = 0

    for border in range(band_size, search_size):
        border_intensity = np.mean(intensity_profile[border - band_size:border])
        image_intensity = np.mean(intensity_profile[border:border + band_size])

        if border_intensity > dark_limit:
            continue
        if image_intensity - border_intensity < minimum_contrast:
            continue
        if gradient_profile[border] > best_score:
            best_score = gradient_profile[border]
            best_border = border

    return best_border


def find_end_border(
    gradient_profile,
    intensity_profile,
    search_size,
    band_size,
    dark_limit,
    minimum_contrast,
):
    length = intensity_profile.size
    best_score = -np.inf
    best_border = length

    for border in range(length - search_size, length - band_size):
        image_intensity = np.mean(intensity_profile[border - band_size:border])
        border_intensity = np.mean(intensity_profile[border:border + band_size])

        if border_intensity > dark_limit:
            continue
        if image_intensity - border_intensity < minimum_contrast:
            continue
        if gradient_profile[border] > best_score:
            best_score = gradient_profile[border]
            best_border = border

    return best_border


def crop_black_border(inpu, max_border_fraction=0.2):
    height, width = inpu.shape[:2]
    if inpu.ndim == 3:
        gray = np.mean(inpu, axis=2)
    else:
        gray = inpu

    gradient_x, gradient_y, _ = image_gradient.image_gradient(gray, method="sobel")

    row_margin = int(width * 0.1)
    col_margin = int(height * 0.1)
    center_cols = slice(row_margin, width - row_margin)
    center_rows = slice(col_margin, height - col_margin)

    row_gradient = np.mean(np.abs(gradient_y[:, center_cols]), axis=1)
    col_gradient = np.mean(np.abs(gradient_x[center_rows, :]), axis=0)
    row_intensity = np.mean(gray[:, center_cols], axis=1)
    col_intensity = np.mean(gray[center_rows, :], axis=0)

    row_gradient = smooth_profile(row_gradient)
    col_gradient = smooth_profile(col_gradient)

    low = np.percentile(gray, 5)
    high = np.percentile(gray, 95)
    intensity_range = high - low
    if intensity_range <= 0:
        return inpu

    dark_limit = low + intensity_range * 0.25
    minimum_contrast = intensity_range * 0.05
    row_search_size = max(2, int(height * max_border_fraction))
    col_search_size = max(2, int(width * max_border_fraction))
    row_band_size = max(2, int(height * 0.01))
    col_band_size = max(2, int(width * 0.01))

    top = find_start_border(
        row_gradient,
        row_intensity,
        row_search_size,
        row_band_size,
        dark_limit,
        minimum_contrast,
    )
    bottom = find_end_border(
        row_gradient,
        row_intensity,
        row_search_size,
        row_band_size,
        dark_limit,
        minimum_contrast,
    )
    left = find_start_border(
        col_gradient,
        col_intensity,
        col_search_size,
        col_band_size,
        dark_limit,
        minimum_contrast,
    )
    right = find_end_border(
        col_gradient,
        col_intensity,
        col_search_size,
        col_band_size,
        dark_limit,
        minimum_contrast,
    )

    if left >= right or top >= bottom:
        return inpu

    return inpu[top:bottom, left:right]
