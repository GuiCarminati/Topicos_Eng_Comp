import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------- Produto Matricial -----------------------
# #a = [[1,2],[2,4]]
#
# a = np.array([[1,2],[2,4]])    # Usando Numpy
# b = np.array([[3,2],[4,1]])
#
# AM = a*b
# PM = np.matmul(a,b)
# #PM = a.dot(b)                  # Só funciona utilizando numpy
#
# print("AM = \n", AM, "\n")
# print("PM = \n", PM)
# -------------------------------- Ajuste de Brilho e Contraste ---------------------------

import numpy as np
import cv2

img = cv2.imread('lena_color.png', cv2.IMREAD_COLOR)
g = np.zeros(img.shape, img.dtype)                              # Cria matriz de saída
alfa = 2.2                                                      # Controle de contraste (Ganho)
beta = 50                                                       # Controle de brilho (Bias)
for x in range(img.shape[0]):                                   # Linha
    for y in range(img.shape[1]):                               # Coluna
        for c in range(img.shape[2]):                           # Canais
            g[x,y,c] = np.clip(alfa*img[x,y,c] + beta, 0, 255)  # Limita os valores de saída

cv2.imshow("Original ", img)
cv2.imshow("Modificada",g)
cv2.waitKey(0)



# plt.figure(1)
# plt.imshow(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))
#
#
#
# plt.show()
