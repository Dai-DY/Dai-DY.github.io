import numpy as np
import cv2
import time
import scipy
from src.conv import conv_2d_2loops, conv_2d_4loops

kernel = np.array([
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1], 
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1]])
kernel = kernel / np.sum(kernel)

Dx = np.array([[1, 0, -1]])
Dy = np.transpose(np.array([[1, 0, -1]]))


img = cv2.imread("pic/selfi.jpg", cv2.IMREAD_GRAYSCALE)
Mode="same"

start = time.perf_counter()
oupu4 = conv_2d_4loops(img, kernel, Mode)
elapsed = time.perf_counter() - start
print(f"Output shape = {oupu4.shape}")
print(f"Elapsed time for 4 loop = {elapsed}")

print("-"*50)

start = time.perf_counter()
oupu2 = conv_2d_2loops(img, kernel, Mode)
elapsed = time.perf_counter() - start
print(f"Output shape = {oupu2.shape}")
print(f"Elapsed time for 2 loop = {elapsed}")

print("-"*50)

start = time.perf_counter()
oupusc = scipy.signal.convolve2d(img, kernel, mode='same')
elapsed = time.perf_counter() - start
print(f"Output shape = {oupusc.shape}")
print(f"Elapsed time for scipy.signal.convolve2d = {elapsed}")

oupu_show = np.clip(oupu2, 0, 255).astype(np.uint8)

Gx = scipy.signal.convolve2d(oupu2, Dx, mode='same')
Gy = scipy.signal.convolve2d(oupu2, Dy, mode='same')

max_abs = max(np.max(np.abs(Gx)), np.max(np.abs(Gy)))

Gx_show = np.clip(127.5 + 127.5 * Gx / max_abs, 0, 255).astype(np.uint8)
Gy_show = np.clip(127.5 + 127.5 * Gy / max_abs, 0, 255).astype(np.uint8)

cv2.imshow("Box filter", oupu_show)
cv2.imshow("Partial derivative Gx", Gx_show)
cv2.imshow("Partial derivative Gy", Gy_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
