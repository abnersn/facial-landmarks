import pickle
import numpy as np
import cv2
from modules.util import plot

with open('model.data', 'rb') as f:
    model = pickle.load(f)


NUMBER_OF_PARAMS = 120

def n(a):
    pass

for coord in ['X', 'Y']:
    cv2.namedWindow(coord)
    for i in range(NUMBER_OF_PARAMS):
        cv2.createTrackbar('{}'.format(i), coord, 50, 100, n)

while True:
    image = np.zeros([500, 500, 3], dtype=np.uint8)
    vector = []
    for i in range(NUMBER_OF_PARAMS):
        x_pos = cv2.getTrackbarPos('{}'.format(i), 'X')
        y_pos = cv2.getTrackbarPos('{}'.format(i), 'Y')
        vector.append([x_pos, y_pos])
    vector = (np.array(vector) - 50) / 100 * 2

    shape = model.deform(vector) * 100 + [250, 250]
    plot(image, shape)

    cv2.imshow('Image', image)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break