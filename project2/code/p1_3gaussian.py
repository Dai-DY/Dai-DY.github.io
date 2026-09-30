import numpy as np
import cv2
import scipy
from pathlib import Path

img = cv2.imread("pic/cameraman.png", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32) / 255.0

Dx = np.array([[1, 0, -1]])
Dy = np.transpose(np.array([[1, 0, -1]]))

g = cv2.getGaussianKernel(9, 1.5)
G = g @ g.T

img_blur = scipy.signal.convolve2d(img, G, mode="same")

Gx = scipy.signal.convolve2d(img_blur, Dx, mode="same")
Gy = scipy.signal.convolve2d(img_blur, Dy, mode="same")

gradient = np.sqrt(Gx ** 2 + Gy ** 2)

threshold = 0.15
binary_gradient = np.where(gradient >= threshold, 255, 0).astype(np.uint8)

DoG_x = scipy.signal.convolve2d(G, Dx, mode="full")
DoG_y = scipy.signal.convolve2d(G, Dy, mode="full")

G_show = cv2.normalize(G, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
DoG_x_show = cv2.normalize(DoG_x, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
DoG_y_show = cv2.normalize(DoG_y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

kernel_scale = 40
G_show = cv2.resize(G_show,None, fx=kernel_scale, fy=kernel_scale, interpolation=cv2.INTER_NEAREST)
DoG_x_show = cv2.resize(DoG_x_show, None,fx=kernel_scale, fy=kernel_scale, interpolation=cv2.INTER_NEAREST)
DoG_y_show = cv2.resize(DoG_y_show, None,fx=kernel_scale, fy=kernel_scale, interpolation=cv2.INTER_NEAREST)

Gx_DoG = scipy.signal.convolve2d(img, DoG_x, mode="same")
Gy_DoG = scipy.signal.convolve2d(img, DoG_y, mode="same")

gradient_DoG = np.sqrt(Gx_DoG ** 2 + Gy_DoG ** 2)

binary_gradient_DoG = np.where(gradient_DoG >= threshold, 255, 0).astype(np.uint8)

border = max(
    DoG_x.shape[0],
    DoG_x.shape[1],
    DoG_y.shape[0],
    DoG_y.shape[1]
) // 2

gradient_inner = gradient[border:-border, border:-border]
gradient_DoG_inner = gradient_DoG[border:-border, border:-border]
difference = np.abs(gradient_inner - gradient_DoG_inner)

print(f"Maximum interior difference = {np.max(difference)}")

cv2.imshow("Blur img", img_blur)
cv2.imshow("Blur gradient", gradient)
cv2.imshow("Blur gradient binary", binary_gradient)
cv2.imshow("Gaussian kernel", G_show)
cv2.imshow("DoG_x", DoG_x_show)
cv2.imshow("DoG_y", DoG_y_show)
cv2.imshow("DoG gradient binary", binary_gradient_DoG)
cv2.waitKey(0)
cv2.destroyAllWindows()
