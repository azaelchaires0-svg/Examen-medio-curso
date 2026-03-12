import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagenes/imagen.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.hist(gray.ravel(),256,[0,256])
plt.title("Histograma original")
plt.show()

brillo = cv2.convertScaleAbs(gray,alpha=1,beta=40)

contraste = cv2.convertScaleAbs(gray,alpha=1.5,beta=0)

equalized = cv2.equalizeHist(gray)

plt.subplot(1,4,1)
plt.imshow(gray,cmap='gray')
plt.title("Original")

plt.subplot(1,4,2)
plt.imshow(brillo,cmap='gray')
plt.title("Brillo")

plt.subplot(1,4,3)
plt.imshow(contraste,cmap='gray')
plt.title("Contraste")

plt.subplot(1,4,4)
plt.imshow(equalized,cmap='gray')
plt.title("Ecualizacion")

plt.show()
