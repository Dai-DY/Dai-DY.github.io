import matplotlib.pyplot as plt
from src.align_image_code import align_images
import cv2
import numpy as np

def log_spectrum(image):
    shifted_fft = np.fft.fftshift(np.fft.fft2(image))
    spectrum = np.log1p(np.abs(shifted_fft))
    return cv2.normalize(spectrum, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).astype(np.float32)


def gray_to_rgb(image):
    return np.repeat(image[..., np.newaxis], 3, axis=2)


# First load images

# low sf
im1 = plt.imread('pic/bright.jpeg') / 255.

# high sf
im2 = plt.imread('pic/dark.jpeg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

gray1 = cv2.cvtColor(im1_aligned.astype(np.float32),cv2.COLOR_RGB2GRAY,)
gray2 = cv2.cvtColor(im2_aligned.astype(np.float32),cv2.COLOR_RGB2GRAY,)

spectrum1 = log_spectrum(gray1)
spectrum2 = log_spectrum(gray2)

sigma_low = 3
sigma_high = 1.5

color1_lowpass = cv2.GaussianBlur(
    im1_aligned.astype(np.float32),
    ksize=(0, 0),
    sigmaX=sigma_low,
    sigmaY=sigma_low,
    borderType=cv2.BORDER_REFLECT,
)
color2_highpass = im2_aligned.astype(np.float32) - cv2.GaussianBlur(
    im2_aligned.astype(np.float32),
    ksize=(0, 0),
    sigmaX=sigma_high,
    sigmaY=sigma_high,
    borderType=cv2.BORDER_REFLECT,
)

gray1_lowpass = cv2.GaussianBlur(
    gray1,
    ksize=(0, 0),
    sigmaX=sigma_low,
    sigmaY=sigma_low,
    borderType=cv2.BORDER_REFLECT,
)

gray2_highpass = gray2 - cv2.GaussianBlur(
    gray2,
    ksize=(0, 0),
    sigmaX=sigma_high,
    sigmaY=sigma_high,
    borderType=cv2.BORDER_REFLECT,
)

spectrum1_lowpass = log_spectrum(gray1_lowpass)
spectrum2_highpass = log_spectrum(gray2_highpass)

color_mode = "low"

if color_mode == "both":
    low_component = color1_lowpass
    high_component = color2_highpass
elif color_mode == "low":
    low_component = color1_lowpass
    high_component = gray_to_rgb(gray2_highpass)
elif color_mode == "high":
    low_component = gray_to_rgb(gray1_lowpass)
    high_component = color2_highpass
else:
    low_component = gray_to_rgb(gray1_lowpass)
    high_component = gray_to_rgb(gray2_highpass)

print(np.max(low_component + high_component))
output_show = np.clip(low_component + high_component, 0, 1)

output_show = cv2.cvtColor(output_show.astype(np.float32), cv2.COLOR_RGB2BGR)

cv2.imshow("spectrum for low frequency", spectrum1)
cv2.imshow("spectrum for lowpass", spectrum1_lowpass)
cv2.imshow("spectrum for high frequency", spectrum2)
cv2.imshow("spectrum for highpass", spectrum2_highpass)
cv2.imshow("Output", output_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
