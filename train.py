#!/bin/python3.6
import pickle
import os
import numpy as np
import cv2
from time import time
from scipy.spatial.distance import cdist as distance
from modules.data_manager import read_dataset
from modules.procrustes import find_theta, rotate
from multiprocessing import Pool

IMAGE_PATH = './img'
DATA_PATH = './data'
MEAN_SHAPE_PATH = 'mean_shape.data'
REF_POINTS_PATH = 'reference_points.data'
NUMBER_OF_TREES = 500
TREES_DEPTH = 5
NUMBER_OF_REGRESSORS = 1

# COLORS
RED = [0, 0, 255]
WHITE = [255, 255, 255]


def draw_shape(image, shape):
    radius = int(image.shape[0] * 0.005)
    for point in shape:
        draw_point = tuple(np.array(point).astype(int))
        cv2.circle(image, draw_point, radius, WHITE, thickness=-1)


def sort_points(quantity, centroid, radius):
    angles_base = np.full([quantity, 1], 2 * np.pi)
    angles_random = np.random.rand(quantity, 1)
    angles = np.multiply(angles_base, angles_random)
    x_coords = (np.multiply(np.cos(angles),
                            np.random.rand(quantity, 1) * radius))
    y_coords = (np.multiply(np.sin(angles),
                            np.random.rand(quantity, 1) * radius))
    return np.concatenate((x_coords, y_coords), axis=1) + centroid


def similarity_transform(shape_a, shape_b):
    translation_matrix = (np.mean(shape_a, axis=0) - np.mean(shape_b, axis=0))
    shape_a = shape_a - translation_matrix

    product = np.multiply(shape_a, shape_b)
    scale_factor = (np.sum(product) / np.sum(np.power(shape_a, 2)))
    scaled_shape = shape_a * scale_factor

    rotation_angle = find_theta(shape_b, scaled_shape)

    return (scale_factor, rotation_angle)


def warp_points(points, shape_a, shape_b):
    scale, angle = similarity_transform(shape_b, shape_a)
    warped = np.zeros(points.shape)
    distances = distance(points, shape_a)
    for i in range(len(points)):
        closest_point = np.argmin(distances[i])
        offset = points[i] - shape_a[closest_point]
        offset = rotate(offset / scale, -angle)
        warped[i] = shape_b[closest_point] + offset
    return warped


def test(file_name):
    real_shape = np.array(dataset[file_name[:-4]])
    return warp_points(reference_points, mean_shape, real_shape)


def get_mean_shape():
    with open(MEAN_SHAPE_PATH, 'rb') as f:
        return pickle.load(f)


def get_ref_points():
    with open(REF_POINTS_PATH, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    dataset = read_dataset(DATA_PATH)

    mean_shape = get_mean_shape()
    reference_points = get_ref_points()

    start_time = time()
    print('start')
    p = Pool(4)
    data = p.map(test, os.listdir(IMAGE_PATH))

print('ellapsed time: {}'.format(time() - start_time))
