import numpy as np
import cv2
from pathlib import Path


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


def save_image(image, output_path):
    image = np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def save_stack(gaussian, laplacian, image_name):
    gaussian_dir = Path("media") / image_name / "gaussian"
    laplacian_dir = Path("media") / image_name / "laplacian"

    gaussian_dir.mkdir(parents=True, exist_ok=True)
    laplacian_dir.mkdir(parents=True, exist_ok=True)

    for level in range(gaussian.shape[0]):
        save_image(
            gaussian[level],
            gaussian_dir / f"level_{level}.png",
        )

        if level < laplacian.shape[0] - 1:
            max_abs = float(np.max(np.abs(laplacian[level])))
            if max_abs == 0:
                max_abs = 1.0
            laplacian_show = np.clip(0.5 + laplacian[level] / (2 * max_abs),0.0,1.0)
        else:
            laplacian_show = laplacian[level]

        save_image(
            laplacian_show,
            laplacian_dir / f"level_{level}.png",
        )


image_a = cv2.imread("pic/apple.jpeg", cv2.IMREAD_COLOR)
image_b = cv2.imread("pic/orange.jpeg", cv2.IMREAD_COLOR)

image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

if image_a.shape != image_b.shape:
    image_a = cv2.resize(
        image_a,
        (image_b.shape[1], image_b.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


levels = 6
base_sigma = 2

gaussian_a = gaussian_stack(image_a, levels, base_sigma)
gaussian_b = gaussian_stack(image_b, levels, base_sigma)

laplacian_a = laplacian_stack(gaussian_a)
laplacian_b = laplacian_stack(gaussian_b)

save_stack(gaussian_a, laplacian_a, "apple")
save_stack(gaussian_b, laplacian_b, "orange")
