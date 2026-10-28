# Project 4 · Part A

Open `index.html` through the repository's normal static preview. The portfolio homepage links to `./project4/`.

The report follows the [Fall 2026 Part A assignment](https://cal-cs180.github.io/fa26/hw/proj3/partA.html) and uses the same colors, typography, navigation, image lightbox, and responsive layout as Projects 2 and 3.

## Available material

- All eleven outdoor source photographs, three Gateway photographs, eleven Frontview photographs, and the updated desk, book, and bottle views used for rectification.
- The supplied `output/proj4_outside11_panorama.jpg`, resized to 2800 pixels wide for display.
- Three two-image alignment tests—Outside, the middle/right Gateway pair, and the middle/right Frontview pair—with manually marked correspondences, recovered homographies, and per-point reprojection errors.
- Monitor, book, and bottle rectification, each with nearest-neighbor and bilinear outputs. The book's inner white panel and the bottle's white label are mapped from oblique quadrilaterals to upright rectangles using the original inverse samplers.
- Three three-frame blending examples for Outside, Gateway, and Frontview. Each aligns consecutive photographs to the middle image and includes the averaged aligned inputs, middle-reference placement, three-way hard/soft masks, hard overwrite, and two-band blend.
- A three-frame Gateway panorama, using frame 2 as the planar reference.
- A 3200 × 1390 Frontview panorama assembled from frames 1–10, using frame 6 as the planar reference.

The original project directory `/home/daidy/workspace/CS180/proj4` was read without changing its code, images, outputs, or existing bytecode. Additional image experiments were run from separate temporary drivers and did not use automatic interest-point detection.

## New alignment and blending

Both former mosaic placeholders have been replaced with actual sources and results. The monitor source visualization is cropped around the display and desk; the book and bottle examples record their corrected white-region corners, target rectangles, homographies, and both sampling results.

New landmarks were seeded manually. Local normalized-correlation searches refine the target locations around those seeds; a fixed-seed reprojection-consistency filter removes mismatches and points with incompatible parallax. Final retained pairs are fitted by least squares.

Gateway and Frontview use adjacent-image planar homographies fitted with the same least-squares formulation as the original `computH` function. The transforms are composed into their middle reference views. The final Frontview mosaic omits frame 11 to reduce the most severe right-edge stretching while retaining the characteristic irregular boundary of a planar panorama.

New panoramas compose transforms into a common middle view and sample each original once with inverse bilinear resampling. They preserve valid-pixel masks and blend a Gaussian low band and residual high band using masks derived from distance to the warped-image boundaries.

The balcony railing and flower pots retain parallax seams. Moving pedestrians and the foreground path also leave local differences in Gateway. These limitations are described beside the results.

The desk monitor and book are approximately planar. The bottle label lies on a curved surface, so its homography matches the selected boundary without modeling every interior deformation. The source point coordinates for the supplied eleven-frame panorama were not available.

## Measurements and provenance

`data/manual-points.json` records all new manual point coordinates, target rectangles, homographies, crop locations, reprojection errors, single-run sampling times, and hashes of the original Python files. Coordinates refer to the original loader's resized 1200 × 900 images, not to the full-resolution JPEGs.

`data/new-panorama-points.json` records the manual seeds, local patch refinements, retained/rejected pairs, adjacent transforms, canvas bounds, composed transforms, projection settings, exposure adjustments, and source-image hashes for Gateway and Frontview. The page visualizes representative point pairs from these experiments.

The report uses the existing MathJax CDN for formulas. All photographic assets, CSS, JavaScript, and point data are local to this project directory.
