# Project 1: Images of the Russian Empire

This project reconstructs a color image from a glass plate scan containing three grayscale channels stacked vertically in **B (blue), G (green), R (red)** order. The green and red channels are aligned to the blue channel, followed by cropping, white balance, and contrast enhancement.

## 1. Directory Structure and Modules

```text
project1/
├── index.html                 # Project report and results gallery
├── media/                     # Images and pyramid visualization assets for the report
└── code/
    ├── README.md
    ├── colorize_skel.py        # Entry point for processing a single image
    ├── src/
    │   ├── simple_align.py     # Single-scale translation search and scoring
    │   ├── coarse_to_fine.py   # Image pyramid and multiscale alignment
    │   ├── image_gradient.py   # Image gradient computation
    │   ├── image_crop.py       # Valid-overlap and dark-border cropping
    │   ├── white_balance.py    # Gray World white balance
    │   └── image_contrast.py   # Automatic contrast stretching
    ├── data/                  # Create before running; stores original plate scans
    └── output/                # Create before running; stores reconstructed images
```

### Entry Point: `colorize_skel.py`

The main script runs the complete processing pipeline:

1. Read the input as a grayscale image using OpenCV and convert it to a floating-point array in `[0, 1]`.
2. Split the image into three equal-height B, G, and R channels, discarding any remaining bottom rows.
3. Align G and R to B using pyramid NCC alignment with Sobel gradients by default.
4. Stack the aligned channels in R, G, B order to create a color image.
5. Apply common-overlap cropping, dark-border cropping, Gray World white balance, and contrast stretching, in that order.
6. Display a preview, then convert the result to OpenCV's BGR format and save it as a JPEG.

### Processing Modules

| Module | Main Functions | Purpose |
| --- | --- | --- |
| `simple_align.py` | `auto_crop`, `l2_score`, `ncc_score`, `align_l2`, `align_NCC` | Search candidate translations and score corresponding overlapping regions. By default, an additional 10% is removed from each edge before scoring. The L2 implementation uses mean squared error, where lower is better; normalized cross-correlation (NCC) is better when higher. The default search radius is ±15 pixels around the search center. |
| `coarse_to_fine.py` | `downsample`, `align_NCC`, `pyramid_align` | Repeatedly downsample by a factor of 2 until the longest side is smaller than `min_size=300`, then search within ±30 pixels. At each finer level, double the previous offset and refine within ±5 pixels. NCC can operate on grayscale intensities or gradient magnitudes. |
| `image_gradient.py` | `filter_2d`, `image_gradient`, `sobel`, `prewitt`, `roberts` | Implement 2D filtering with NumPy and return horizontal gradients, vertical gradients, and gradient magnitudes. Support Sobel, Prewitt, and Roberts operators, with Sobel as the default. |
| `image_crop.py` | `crop_common_overlap`, `crop_black_border` | Use channel offsets to crop the common valid region and remove pixels wrapped by `np.roll`. Then detect dark borders using Sobel gradients, intensity changes, and smoothed row/column profiles within the outer 20% of each dimension by default. |
| `white_balance.py` | `gray_world` | Adjust channel gains toward the average of the three channel means to reduce an overall color cast. |
| `image_contrast.py` | `automatic_contrast` | Use the global minimum and maximum across the entire image to linearly stretch intensities to `[0, 1]`. Return an all-zero array for a constant input image. |

## 2. Running the Code

### Step 1: Enter the Code Directory

From the repository root, run:

```bash
cd project1/code
```

Run the following commands from this directory. The script resolves `data/` and `output/` relative to the **current working directory**.

### Step 2: Run the Script

```bash
python colorize_skel.py
```

The script prints the best score from each NCC search and the final channel offsets:

```text
G_offset=(dx, dy) R_offset=(dx, dy)
```

It then displays a preview of the reconstructed image. **Close the preview window to let the script continue to the saving step.** The output path is:

```text
output/<input_filename_without_extension>_out.jpg
```

## 3. Changing the Alignment Method

Replace the two G/R alignment calls in `colorize_skel.py` to select another method.

### Pyramid Alignment (Default)

```python
ag, offsetg = coarse_to_fine.pyramid_align(g, b, gradient_method="sobel")
ar, offsetr = coarse_to_fine.pyramid_align(r, b, gradient_method="sobel")
```

Set `gradient_method` to `"sobel"`, `"prewitt"`, or `"roberts"`. Set it to Python's `None` to compute NCC directly on grayscale intensities. The `min_size` parameter controls the size threshold for stopping downsampling.
