import numpy as np
import cv2
import os

src = "./visualization/front+label"
imgs = os.listdir(src)

n = int(len(imgs)**0.5)
w = 960
h = 720

print(n)

IMG = np.zeros((h*n,w*n,3))

def merge(src,imgs,IMG):
	for i,img in enumerate(imgs):
		image = cv2.imread(os.path.join(src,img))
		#print(image.shape)
		#print(i//n," ",i%n)
		IMG[(i//n)*h:(i//n+1)*h,(i%n)*w:(i%n+1)*w,:] = image

	return IMG

merge_image = merge(src,imgs,IMG)
cv2.imwrite("merge_front+label.jpg",merge_image)		

