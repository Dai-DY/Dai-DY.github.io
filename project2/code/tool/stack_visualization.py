"""Visualization helpers for Gaussian/Laplacian stack blending."""

import matplotlib.pyplot as plt
import numpy as np


def _show_image(axis, image):
    if image.ndim == 2:
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        axis.imshow(np.clip(image, 0, 1))
    axis.axis("off")


def _create_final_result_figure(result):
    """Create a dedicated window for the final blended image."""
    result_figure, result_axis = plt.subplots(
        num="Final Blended Result",
        figsize=(10, 8),
    )
    _show_image(result_axis, result)
    result_axis.set_title("Final Blended Result")
    result_figure.tight_layout()


def visualize_blend_result(image_a, image_b, mask, result, blend_data):
    """Show inputs and every level of a multiresolution blend."""
    gaussian_mask = blend_data["gaussian_mask"]
    masked_a = blend_data["masked_a"]
    masked_b = blend_data["masked_b"]
    blended_laplacian = blend_data["blended_laplacian"]
    levels = blended_laplacian.shape[0]

    fig, axes = plt.subplots(
        levels + 1,
        4,
        figsize=(16, 3 * (levels + 1)),
        squeeze=False,
    )

    first_row = (
        (image_a, "Image A"),
        (image_b, "Image B"),
        (mask, "Original Mask"),
        (result, "Final Blended Result"),
    )
    for column, (image, title) in enumerate(first_row):
        _show_image(axes[0, column], image)
        axes[0, column].set_title(title)

    axes[0, 0].text(
        -0.08,
        0.5,
        "Inputs",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="center",
        rotation=90,
    )

    column_titles = (
        "Masked A Component",
        "Masked B Component",
        "Gaussian Mask",
        "Blended Component",
    )

    for level in range(levels):
        row = level + 1
        components = (
            masked_a[level],
            masked_b[level],
            blended_laplacian[level],
        )

        if level < levels - 1:
            # Signed Laplacian bands share a contrast scale at each level.
            max_abs = max(
                float(np.percentile(np.abs(component), 99))
                for component in components
            )
            if max_abs == 0:
                max_abs = 1.0

            displayed_a = np.clip(0.5 + masked_a[level] / (2 * max_abs), 0, 1)
            displayed_b = np.clip(0.5 + masked_b[level] / (2 * max_abs), 0, 1)
            displayed_blend = np.clip(
                0.5 + blended_laplacian[level] / (2 * max_abs),
                0,
                1,
            )
        else:
            # The last layer is a low-frequency residual, not a signed band.
            displayed_a = np.clip(masked_a[level], 0, 1)
            displayed_b = np.clip(masked_b[level], 0, 1)
            displayed_blend = np.clip(blended_laplacian[level], 0, 1)

        _show_image(axes[row, 0], displayed_a)
        _show_image(axes[row, 1], displayed_b)
        _show_image(axes[row, 2], gaussian_mask[level])
        _show_image(axes[row, 3], displayed_blend)

        for column, title in enumerate(column_titles):
            if level == 0:
                axes[row, column].set_title(title)

        level_name = f"Level {level}"
        if level == levels - 1:
            level_name += "\n(residual)"
        axes[row, 0].text(
            -0.08,
            0.5,
            level_name,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
        )

    fig.suptitle("Multiresolution Blend: All Stack Levels", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    # Create this figure before the blocking show() call so GUI backends display
    # the final image in its own window alongside the stack overview.
    _create_final_result_figure(result)
    plt.show()


def visualize_stacks(gaussian, laplacian):
    """Display precomputed Gaussian and Laplacian stacks in two columns."""
    levels = gaussian.shape[0]
    if laplacian.shape[0] != levels:
        raise ValueError("Gaussian and Laplacian stacks must have equal levels")

    fig, axes = plt.subplots(
        levels,
        2,
        figsize=(10, 3 * levels),
        squeeze=False,
    )

    image_min = float(np.min(gaussian[0]))
    image_max = float(np.max(gaussian[0]))

    for level in range(levels):
        axes[level, 0].imshow(
            gaussian[level],
            cmap="gray",
            vmin=image_min,
            vmax=image_max,
        )
        axes[level, 0].set_ylabel(f"Level {level}")
        axes[level, 0].axis("off")

        if level < levels - 1:
            max_abs = float(np.max(np.abs(laplacian[level])))
            if max_abs == 0:
                max_abs = 1.0
            axes[level, 1].imshow(
                laplacian[level],
                cmap="gray",
                vmin=-max_abs,
                vmax=max_abs,
            )
        else:
            axes[level, 1].imshow(
                laplacian[level],
                cmap="gray",
                vmin=image_min,
                vmax=image_max,
            )
        axes[level, 1].axis("off")

    axes[0, 0].set_title("Gaussian Stack")
    axes[0, 1].set_title("Laplacian Stack")
    fig.tight_layout()
    plt.show()
