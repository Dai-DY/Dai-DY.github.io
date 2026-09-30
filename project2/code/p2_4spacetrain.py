import numpy as np
import scipy
import cv2

from tool.stack_visualization import (
    visualize_blend_result,
    visualize_stacks,
)


def gaussian_blur(image, sigma):
    return cv2.GaussianBlur(
        image,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )



def gaussian_stack(image, levels, base_sigma):
    stack = [image.astype(np.float32)]

    for level in range(1, levels):
        level_sigma = base_sigma * (2 ** (level - 1))
        blurred = gaussian_blur(stack[0], level_sigma)
        stack.append(blurred)

    return np.stack(stack, axis=0)

def laplacian_stack(gaussian):
    laplacian = np.empty_like(gaussian)

    for level in range(gaussian.shape[0] - 1):
        laplacian[level] = (
            gaussian[level] - gaussian[level + 1]
        )

    laplacian[-1] = gaussian[-1]
    return laplacian


def multiresolution_blend(image_a, image_b, mask, levels, base_sigma):
    gaussian_a = gaussian_stack(image_a, levels, base_sigma)
    gaussian_b = gaussian_stack(image_b, levels, base_sigma)
    gaussian_mask = gaussian_stack(mask, levels, base_sigma)

    laplacian_a = laplacian_stack(gaussian_a)
    laplacian_b = laplacian_stack(gaussian_b)

    mask_weights = gaussian_mask
    if image_a.ndim == 3:
        mask_weights = mask_weights[..., np.newaxis]

    masked_a = mask_weights * laplacian_a
    masked_b = (1.0 - mask_weights) * laplacian_b
    blended_laplacian = masked_a + masked_b

    result = np.sum(blended_laplacian, axis=0)
    blend_data = {
        "gaussian_mask": gaussian_mask,
        "laplacian_a": laplacian_a,
        "laplacian_b": laplacian_b,
        "masked_a": masked_a,
        "masked_b": masked_b,
        "blended_laplacian": blended_laplacian,
    }
    return np.clip(result, 0.0, 1.0), blend_data


def main():
    image_a = cv2.imread("pic/space_train_image_a.png", cv2.IMREAD_COLOR)
    image_b = cv2.imread("pic/space.jpg", cv2.IMREAD_COLOR)

    image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if image_a.shape != image_b.shape:
        image_b = cv2.resize(image_b,(image_a.shape[1], image_a.shape[0]),interpolation=cv2.INTER_AREA)

    levels = 6
    base_sigma = 3

    mask = cv2.imread(
        "mask/space_train_mask.png",
        cv2.IMREAD_GRAYSCALE,
    ).astype(np.float32) / 255.0

    result, blend_data = multiresolution_blend(
        image_a,
        image_b,
        mask,
        levels,
        base_sigma,
    )
    visualize_blend_result(image_a, image_b, mask, result, blend_data)


if __name__ == "__main__":
    main()
