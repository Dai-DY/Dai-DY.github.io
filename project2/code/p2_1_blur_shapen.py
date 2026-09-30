import numpy as np
import cv2
import scipy

img = cv2.imread("pic/original_img.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
img = img.astype(np.float32) / 255.0

g = cv2.getGaussianKernel(9, 3)
G = g @ g.T

def unsharp_mask_filter(G_kernel, alpha):
    unit_impulse = np.zeros_like(G_kernel)
    unit_impulse[G_kernel.shape[0] // 2, G_kernel.shape[1] // 2] = 1
    return (1+alpha)*unit_impulse - alpha*G_kernel

sharpening_filter = unsharp_mask_filter(G, 4)

blur_img = []

for i in range(3):
    img_blur_chanel = scipy.signal.convolve2d(img[:,:,i], G, mode="same")
    blur_img.append(img_blur_chanel)

blur_img = np.stack(blur_img, axis=2)

img_sharpen = []
for i in range(3):
    img_sharpen_chanel = scipy.signal.convolve2d(blur_img[:,:,i], sharpening_filter, mode="same")
    img_sharpen.append(img_sharpen_chanel)

img_sharpen = np.stack(img_sharpen, axis=2)

cv2.imshow("Origen", img)
cv2.imshow("Blur", blur_img)
cv2.imshow("Sharpen", img_sharpen)
cv2.waitKey(0)
    
