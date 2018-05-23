#!/usr/bin/python
import pickle
import numpy as np
import cv2
from modules.util import plot
from modules.face_model import ShapeModel
from modules.procrustes import root_mean_square, calculate_procrustes, find_theta, rotate

with open('dev.data', 'rb') as f:
    dataset = pickle.load(f)

NUMBER_OF_PARAMS = 60
START_INDEX = 120
POINTS_TO_REMOVE = int(10 * 194 / 100)

model = ShapeModel(NUMBER_OF_PARAMS, calculate_procrustes(dict(
    [(sample['file_name'], sample['annotation']) for sample in dataset]
)))

sample = dataset[0]

normalized = sample['annotation']
normalized -= np.mean(normalized, axis=0)
normalized /= root_mean_square(normalized)
theta = find_theta(model.base_shape, normalized)
normalized = rotate(normalized, theta)


faulty_points = np.random.randint(0, 194, POINTS_TO_REMOVE)
faulty_points = [i + START_INDEX for i in range(POINTS_TO_REMOVE)]

while True:
    real = np.zeros([400, 400, 3], np.uint8)
    base = np.zeros([400, 400, 3], np.uint8)


    real_shape = np.delete(normalized, faulty_points, axis=0)
    params = model.retrieve_parameters_falty(real_shape, faulty_points)
    
    real_shape *= 50
    real_shape += [200, 200]


    base_shape = model.deform(params)
    base_shape *= 50
    base_shape += [200, 200]


    plot(base, base_shape)
    plot(real, normalized * 50 + [200, 200], [255, 0, 0])
    plot(real, real_shape)

    cv2.imshow('base', base)
    cv2.imshow('real', real)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break



'''
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
'''