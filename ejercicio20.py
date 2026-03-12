import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Captura de pantalla 2026-03-11 200313.png',0)

h,w = img.shape

scale200 = cv2.resize(img,None,fx=2,fy=2)
scale150 = cv2.resize(img,None,fx=1.5,fy=1.5)
scale50 = cv2.resize(img,None,fx=0.5,fy=0.5)

M = np.float32([[1,0,80],[0,1,45]])
traslacion = cv2.warpAffine(img,M,(w,h))

center = (w//2,h//2)
R = cv2.getRotationMatrix2D(center,45,1)

rotacion = cv2.warpAffine(img,R,(w,h))

plt.subplot(231)
plt.imshow(img,cmap='gray')
plt.title("Original")

plt.subplot(232)
plt.imshow(scale200,cmap='gray')
plt.title("200%")

plt.subplot(233)
plt.imshow(scale150,cmap='gray')
plt.title("150%")

plt.subplot(234)
plt.imshow(scale50,cmap='gray')
plt.title("50%")

plt.subplot(235)
plt.imshow(traslacion,cmap='gray')
plt.title("Traslacion")

plt.subplot(236)
plt.imshow(rotacion,cmap='gray')
plt.title("Rotacion")

plt.show()
