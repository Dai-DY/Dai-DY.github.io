import numpy as np

def auto_crop(inpu, reference, dx, dy, border_fraction=0.1):
    height, width = reference.shape
    if abs(dx) >= width or abs(dy) >= height:
        return None, None

    if dy >= 0:
        input_rows = slice(0, height - dy)
        reference_rows = slice(dy, height)
    else:
        input_rows = slice(-dy, height)
        reference_rows = slice(0, height + dy)

    if dx >= 0:
        input_cols = slice(0, width - dx)
        reference_cols = slice(dx, width)
    else:
        input_cols = slice(-dx, width)
        reference_cols = slice(0, width + dx)

    input_cut = inpu[input_rows, input_cols]
    reference_cut = reference[reference_rows, reference_cols]

    # Ignore the outer plate border as well as the non-overlapping area.
    crop_y = int(input_cut.shape[0] * border_fraction)
    crop_x = int(input_cut.shape[1] * border_fraction)
    row_slice = slice(crop_y, -crop_y) if crop_y > 0 else slice(None)
    col_slice = slice(crop_x, -crop_x) if crop_x > 0 else slice(None)

    return input_cut[row_slice, col_slice], reference_cut[row_slice, col_slice]


def l2_score(img1, img2):
    return np.mean((img1.astype(np.float64)-img2.astype(np.float64)) ** 2)

def ncc_score(img1, img2):
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-12:
        return -np.inf
    return np.sum(a * b) / denominator

def align_l2(inpu, reference, center_x=0, center_y=0, max_shift=15):
    best_score = np.inf
    best_offset = (0,0)
    for dy in range(-max_shift+center_y, center_y+max_shift+1):
        for dx in range(-max_shift+center_x, center_x+max_shift+1):
            input_cut, reference_cut = auto_crop(inpu, reference, dx, dy)
            score = l2_score(input_cut, reference_cut)
            if(score < best_score):
                best_score = score
                best_offset = (dx, dy)
    dx, dy = best_offset
    best_image = np.roll(inpu, shift=(dy, dx), axis=(0, 1))
    return best_image, best_offset

def align_NCC(inpu, reference, center_x=0, center_y=0, max_shift=15):
    best_score = -np.inf
    best_offset = (0,0)
    for dy in range(-max_shift+center_y, center_y+max_shift+1):
        for dx in range(-max_shift+center_x, center_x+max_shift+1):
            input_cut, reference_cut = auto_crop(inpu, reference, dx, dy)
            score = ncc_score(input_cut, reference_cut)
            if(score > best_score):
                best_score = score
                best_offset = (dx, dy)
    dx, dy = best_offset
    best_image = np.roll(inpu, shift=(dy, dx), axis=(0, 1))
    print(best_score)
    return best_image, best_offset
