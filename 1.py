from time import sleep
import numpy as np
import cv2

X = np.load('X.npy')
Y = np.load('Y.npy')

print('------------------')
print('X shape:', X.shape)
print()
print('Y shape:', Y.shape)
print('------------------')
print()

k = 3
W = X[0].shape[0] * k
H = X[0].shape[1] * k
cv2.namedWindow('win')
for i in range(1, X.shape[0]):
    y = (X[i]*255).astype('uint8')
    y = cv2.resize(y, (H, W))
    cv2.imshow('win', y)
    if cv2.waitKey(1) >= 0:
        break
    sleep(.1)
cv2.destroyAllWindows()
