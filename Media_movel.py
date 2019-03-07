import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lena_ruido.jpg', 0)

kernel_heigth = 7
kernel_width = 7
kernel = np.array([[1/kernel_width*kernel_heigth]*kernel_width]*kernel_heigth)

heigth_out_img = int(img.shape[0] - (kernel.shape[0]-1))
width_out_img = int(img.shape[1] - (kernel.shape[1]-1))

out_img = np.ndarray([heigth_out_img, width_out_img])

for i in range(out_img.shape[0]):
    for j in range(out_img.shape[1]):
        out_img[i][j] = 0
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                out_img[i][j] += img[i+k][j+l]

plt.figure("ORIGINAL")
plt.imshow(img, cmap='gray')
plt.figure("FILTRADA")
plt.imshow(out_img, cmap='gray')
plt.show()


