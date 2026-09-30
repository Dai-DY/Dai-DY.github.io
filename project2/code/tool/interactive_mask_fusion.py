#!/usr/bin/env python3
"""Select an object manually and generate an irregular mask for image blending."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a foreground region and use GrabCut to generate an irregular mask."
    )
    parser.add_argument("image", type=Path, help="Image containing the object to extract")
    parser.add_argument(
        "--background",
        type=Path,
        help="Optional background image used to create a blended result",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mask_output"),
        help="Output directory (default: mask_output)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of GrabCut iterations (default: 8)",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=9,
        help="Gaussian blur kernel size for feathering; 0 disables it (default: 9)",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def select_object(image: np.ndarray) -> tuple[int, int, int, int]:
    title = "Drag a box around the object, then press ENTER"
    rect = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    x, y, width, height = (int(value) for value in rect)
    if width <= 1 or height <= 1:
        raise RuntimeError("No valid region was selected.")
    return x, y, width, height


def grabcut_labels(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    iterations: int,
) -> np.ndarray:
    labels = np.zeros(image.shape[:2], dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        image,
        labels,
        rect,
        background_model,
        foreground_model,
        max(1, iterations),
        cv2.GC_INIT_WITH_RECT,
    )
    return labels


def labels_to_mask(labels: np.ndarray) -> np.ndarray:
    foreground = np.logical_or(labels == cv2.GC_FGD, labels == cv2.GC_PR_FGD)
    return foreground.astype(np.uint8) * 255


def refine_labels(image: np.ndarray, labels: np.ndarray, iterations: int) -> None:
    has_foreground = np.any(
        np.logical_or(labels == cv2.GC_FGD, labels == cv2.GC_PR_FGD)
    )
    has_background = np.any(
        np.logical_or(labels == cv2.GC_BGD, labels == cv2.GC_PR_BGD)
    )
    if not has_foreground or not has_background:
        return

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        labels,
        None,
        background_model,
        foreground_model,
        max(1, iterations),
        cv2.GC_INIT_WITH_MASK,
    )


def make_preview(
    image: np.ndarray,
    mask: np.ndarray,
    brush_radius: int,
    brush_mode: str,
) -> np.ndarray:
    preview = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255
    overlay = cv2.addWeighted(image, 0.55, green, 0.45, 0)
    preview[mask > 0] = overlay[mask > 0]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 0, 255), 2)
    cv2.putText(
        preview,
        f"Mode: {brush_mode.upper()}   A: add   E: erase   Left-drag: paint",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Brush: {brush_radius}px   [ / ]: resize   U: undo   ENTER: save   R: reselect",
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def edit_mask(
    image: np.ndarray,
    labels: np.ndarray,
    iterations: int,
) -> str:
    title = "Mask preview and brush editor"
    state = {
        "drawing": False,
        "brush_radius": 15,
        "brush_mode": "erase",
        "undo_labels": None,
    }

    def redraw() -> None:
        mask = labels_to_mask(labels)
        preview = make_preview(
            image,
            mask,
            state["brush_radius"],
            state["brush_mode"],
        )
        cv2.imshow(title, preview)

    def paint(event: int, x: int, y: int, flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drawing"] = True
            state["undo_labels"] = labels.copy()

        if state["drawing"] and (
            event == cv2.EVENT_LBUTTONDOWN
            or event == cv2.EVENT_MOUSEMOVE
            or flags & cv2.EVENT_FLAG_LBUTTON
        ):
            label = cv2.GC_FGD if state["brush_mode"] == "add" else cv2.GC_BGD
            cv2.circle(labels, (x, y), state["brush_radius"], label, -1)
            redraw()

        if event == cv2.EVENT_LBUTTONUP:
            state["drawing"] = False
            refine_labels(image, labels, iterations)
            redraw()

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, paint)
    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (10, 13, 32):
            cv2.destroyWindow(title)
            return "save"
        if key in (ord("r"), ord("R")):
            cv2.destroyWindow(title)
            return "reselect"
        if key in (ord("a"), ord("A")):
            state["brush_mode"] = "add"
            redraw()
        if key in (ord("e"), ord("E")):
            state["brush_mode"] = "erase"
            redraw()
        if key in (ord("u"), ord("U")) and state["undo_labels"] is not None:
            labels[:] = state["undo_labels"]
            state["undo_labels"] = None
            redraw()
        if key == ord("["):
            state["brush_radius"] = max(2, state["brush_radius"] - 3)
            redraw()
        if key == ord("]"):
            state["brush_radius"] = min(100, state["brush_radius"] + 3)
            redraw()
        if key == 27:
            cv2.destroyWindow(title)
            return "cancel"


def choose_mask(image: np.ndarray, iterations: int) -> np.ndarray:
    while True:
        rect = select_object(image)
        labels = grabcut_labels(image, rect, iterations)
        action = edit_mask(image, labels, iterations)
        if action == "save":
            cv2.destroyAllWindows()
            return labels_to_mask(labels)
        if action == "cancel":
            cv2.destroyAllWindows()
            raise KeyboardInterrupt


def feather_alpha(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    alpha = mask.astype(np.float32) / 255.0
    if kernel_size > 0:
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        alpha = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
    return alpha[:, :, None]


def blend_with_background(
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    feather: int,
) -> np.ndarray:
    if background.shape[:2] != foreground.shape[:2]:
        background = cv2.resize(
            background,
            (foreground.shape[1], foreground.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    alpha = feather_alpha(mask, feather)
    result = foreground.astype(np.float32) * alpha
    result += background.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def save_outputs(
    image: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    background: np.ndarray | None,
    feather: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "mask.png"), mask)

    cutout = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    cutout[:, :, 3] = mask
    cv2.imwrite(str(output_dir / "cutout.png"), cutout)

    if background is not None:
        blended = blend_with_background(image, background, mask, feather)
        cv2.imwrite(str(output_dir / "blended.png"), blended)


def main() -> None:
    args = parse_args()
    image = read_image(args.image)
    background = read_image(args.background) if args.background else None

    try:
        mask = choose_mask(image, args.iterations)
    except KeyboardInterrupt:
        print("Cancelled. No files were saved.")
        return

    save_outputs(image, mask, args.output_dir, background, args.feather)
    print(f"Mask and transparent cutout saved to: {args.output_dir.resolve()}")
    if background is not None:
        print(f"Blended result saved to: {(args.output_dir / 'blended.png').resolve()}")


if __name__ == "__main__":
    main()
