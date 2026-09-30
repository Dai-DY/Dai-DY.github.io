import numpy as np
import cv2
import time
import scipy

def conv_2d_4loops(img, kernel, Mode = "same"):
    h_img, w_img = img.shape
    print(f"Input shape = {img.shape}")

    h_kernel, w_kernel = kernel.shape

    k = (w_kernel - 1) // 2
    print(f"Kernel size = {h_kernel} * {w_kernel}")

    kernel_f = np.flip(kernel, axis=(0,1))

    w_pad = 0
    h_pad = 0

    if(Mode == "same"):
        print("Mode = same")
        w_pad = (w_kernel - 1) // 2
        h_pad = (h_kernel - 1) // 2

    elif(Mode == "full"):
        print("Mode = full")
        w_pad = w_kernel - 1
        h_pad = h_kernel - 1

    else:
        print("Mode = valid")

    pad_img = np.pad(img, pad_width=((h_pad, h_pad), (w_pad, w_pad)), mode="constant", constant_values=0)

    h_padimg, w_padimg = pad_img.shape

    h_out = h_padimg - h_kernel + 1
    w_out = w_padimg - w_kernel + 1

    out = np.zeros((h_out, w_out))

    for x in range(w_out):
        for y in range(h_out):
            for v in range(h_kernel):
                for u in range(w_kernel):
                    out[y, x] += kernel_f[v, u] * pad_img[(y+v), (x+u)]

    return out

def conv_2d_2loops(img, kernel, Mode = "same"):
    h_img, w_img = img.shape
    print(f"Input shape = {img.shape}")

    h_kernel, w_kernel = kernel.shape

    k = (w_kernel - 1) // 2
    print(f"Kernel size = {h_kernel} * {w_kernel}")

    kernel_f = np.flip(kernel, axis=(0,1))

    w_pad = 0
    h_pad = 0

    if(Mode == "same"):
        print("Mode = same")
        w_pad = (w_kernel - 1) // 2
        h_pad = (h_kernel - 1) // 2

    elif(Mode == "full"):
        print("Mode = full")
        w_pad = w_kernel - 1
        h_pad = h_kernel - 1

    else:
        print("Mode = valid")

    pad_img = np.pad(img, pad_width=((h_pad, h_pad), (w_pad, w_pad)), mode="constant", constant_values=0)

    h_padimg, w_padimg = pad_img.shape

    h_out = h_padimg - h_kernel + 1
    w_out = w_padimg - w_kernel + 1

    out = np.zeros((h_out, w_out))

    for x in range(w_out):
        for y in range(h_out):
            target = pad_img[y:y+h_kernel, x:x+w_kernel]
            out[y, x] = np.sum(target * kernel_f)
    return out
