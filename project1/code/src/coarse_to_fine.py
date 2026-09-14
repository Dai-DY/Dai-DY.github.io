import numpy as np
import skimage as sk

from src import image_gradient
from src import simple_align

def downsample(channel):
    return sk.transform.rescale(
        channel,
        scale=0.5,
        anti_aliasing=True,
        preserve_range=True,
        channel_axis=None,
    )


def align_NCC(inpu, reference, center_x=0, center_y=0, max_shift=15, gradient_method="sobel"):
    if gradient_method is None:
        return simple_align.align_NCC(inpu, reference, center_x, center_y, max_shift)

    _, _, inpu_gradient = image_gradient.image_gradient(inpu, method=gradient_method)
    _, _, reference_gradient = image_gradient.image_gradient(reference, method=gradient_method)
    _, best_offset = simple_align.align_NCC(inpu_gradient, reference_gradient, center_x, center_y, max_shift)

    dx, dy = best_offset
    best_image = np.roll(inpu, shift=(dy, dx), axis=(0, 1))
    return best_image, best_offset


def pyramid_align(inpu, reference, min_size=300, gradient_method="sobel"):
    height, width = reference.shape
    # print(height, width)
    if max(height, width) < min_size:
        # print("small!")
        return align_NCC(inpu, reference, max_shift=30, gradient_method=gradient_method)

    downsample_inpu = downsample(inpu)
    downsample_reference = downsample(reference)
    _, offset = pyramid_align(downsample_inpu, downsample_reference, min_size, gradient_method)
    dx, dy = offset
    dx *= 2
    dy *= 2

    return align_NCC(inpu, reference, dx, dy, max_shift=5, gradient_method=gradient_method)
