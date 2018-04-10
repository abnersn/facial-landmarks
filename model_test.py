#!/bin/python3.6
import pickle, dill
import os, sys
import numpy as np
import cv2, dlib
import argparse
from modules.util import *
from modules.face_model import ShapeModel
from modules.procrustes import calculate_procrustes, root_mean_square

parser = argparse.ArgumentParser(description='Builds a PCA model over a dataset')
parser.add_argument('dataset_path', help='Dataset directory.')
parser.add_argument('--parameters', help='Desired number of parameters.', default=12)
parser.add_argument('--points', help='Desired number of sample points per shape point.', default=4)
args = parser.parse_args()

def nothing(x):
    pass

def get_params(number_of_params, params_range=100):
    params = np.zeros([number_of_params, 2])
    for i in range(number_of_params):
        x = cv2.getTrackbarPos('X[{}]'.format(i), 'X Params')
        y = cv2.getTrackbarPos('Y[{}]'.format(i), 'Y Params')
        params[i] = np.array([x, y])
    return (params - 100) / 200 * params_range

with open(args.dataset_path, 'rb') as f:
    dataset = dill.load(f)

for file_name, data in dataset.items():
    dataset[file_name] = data['annotations']

normalized = calculate_procrustes(dataset)
model = ShapeModel(args.parameters, normalized)

cv2.namedWindow('X Params')
cv2.namedWindow('Y Params')
for i in range(model.number_of_params):
    cv2.createTrackbar('X[{}]'.format(i), 'X Params', 100, 200, nothing)
    cv2.createTrackbar('Y[{}]'.format(i), 'Y Params', 100, 200, nothing)

radius = 0.60 * root_mean_square(model.base_shape)
sample_points = np.zeros([len(model.base_shape), args.points, 2])
for i, point in enumerate(model.base_shape):
    sample_points[i] = sort_points(args.points, point, radius)


def warp2(shape_a, shape_b, groups):
    scale, angle, _ = similarity_transform(shape_b, shape_a)
    new_groups = np.zeros(groups.shape)
    for i, group in enumerate(groups):
        for j, point in enumerate(group):
            offset = point - shape_a[i]
            offset = rotate(offset / scale, -angle)
            new_groups[i][j] = shape_b[i] - offset
    return new_groups

while True:
    img = np.zeros((600, 800, 3), np.uint8)
    deform_params = get_params(model.number_of_params, 5)
    shape = model.deform(deform_params)
    samples = warp2(shape, model.base_shape, sample_points).flatten().reshape([args.points * len(model.base_shape), 2])
    shape = shape * 100 + [400, 300]
    samples = samples * 100 + [400, 300]
    plot(img, shape)
    plot(img, samples)
    cv2.imshow('image',img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break