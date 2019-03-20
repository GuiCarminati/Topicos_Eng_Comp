import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv_Gray(img, kernel):

    # Pega tamanho do kernel
    heigth_kernel = kernel.shape[0]
    width_kernel = kernel.shape[1]

    # Redução da imagem de saída
    heigth_out_img = int(img.shape[0] - (heigth_kernel - 1))
    width_out_img = int(img.shape[1] - (width_kernel - 1))

    # Define tamanho imagem de saída
    out_img = np.ndarray([heigth_out_img, width_out_img])

    # Convolução
    for i in range(heigth_out_img):                 # Linhas da imagem
        for j in range(width_out_img):              # Colunas da imagem
            aux = 0
            for k in range(kernel.shape[0]):        # Linhas do kernel
                for l in range(kernel.shape[1]):    # Colunas do kernel
                    aux += img[i+k][j+l] * kernel[k][l]
            out_img[i][j] = np.clip(aux, 0, aux)
    return out_img

# Abertura da imagem com ruído
img = cv2.imread('Imagens/lena_ruido.jpg', 0)

# ------------------- Filtro de Média
# Dimensão do kernel
kernel_heigth = 7
kernel_width = 7
# Definição dos valores do kernel
kernel = np.array([[(1/(kernel_width*kernel_heigth))]*kernel_width]*kernel_heigth)

# Resultado do filtro de média
mean_img = conv_Gray(img, kernel)
print(mean_img)

# ------------------- Detecção de Bordas
kernel_ed = np.array([[-1,  -1, -1],
                      [-1,   8, -1],
                      [-1,  -1, -1]])

ed_original = conv_Gray(img, kernel_ed)
ed_filter = conv_Gray(mean_img, kernel_ed)
print(ed_filter[0])

plt.figure("ORIGINAL")
plt.imshow(img, cmap='gray')
plt.figure("FILTRADA")
plt.imshow(mean_img, cmap='gray')
plt.figure("DETECÇÃO DE BORDA - ORIGINAL")
plt.imshow(ed_original, cmap='gray')
plt.figure("DETECÇÃO DE BORDA - FILTRADA")
plt.imshow(ed_filter, cmap='gray')

plt.show()


