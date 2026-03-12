import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagenes/virus.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

Z = img.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
K = 3

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
clustered = res.reshape((img.shape))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img_rgb)

plt.subplot(1,3,2)
plt.title("Umbral")
plt.imshow(thresh,cmap='gray')

plt.subplot(1,3,3)
plt.title("Clustering")
plt.imshow(clustered)

plt.show()
