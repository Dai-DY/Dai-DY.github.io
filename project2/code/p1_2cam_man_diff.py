import numpy as np
import cv2
import scipy

img = cv2.imread("pic/cameraman.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

Dx = np.array([[1, 0, -1]])
Dy = np.transpose(np.array([[1, 0, -1]]))

Gx = scipy.signal.convolve2d(img, Dx, mode='same')
Gy = scipy.signal.convolve2d(img, Dy, mode='same')

max_abs = max(np.max(np.abs(Gx)), np.max(np.abs(Gy)))


Gx_show = np.clip(127.5 + 127.5 * Gx / max_abs, 0, 255).astype(np.uint8)
Gy_show = np.clip(127.5 + 127.5 * Gy / max_abs, 0, 255).astype(np.uint8)

cv2.imshow("Partial derivative Gx", Gx_show)
cv2.imshow("Partial derivative Gy", Gy_show)

gradient = np.sqrt(Gx ** 2 + Gy ** 2)

threshold = 0.29

binary_gradient = np.where(gradient >= threshold, 255, 0).astype(np.uint8)

cv2.imshow("Gradient", gradient)
cv2.imshow("Binary gradient", binary_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
