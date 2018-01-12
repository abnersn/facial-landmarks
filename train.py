#!/bin/python3.6
from multiprocessing import Pool
from time import time
import pickle
import os
import numpy as np
import cv2
from imutils import resize
from scipy.spatial.distance import cdist as distance
from modules.data_manager import read_dataset
from modules.procrustes import find_theta, rotate

IMAGE_PATH = './img'
DATA_PATH = './data'
SHAPES_MEAN_PATH = 'shapes_mean.data'
REF_POINTS_PATH = 'ref_points.data'
NUMBER_OF_TREES = 500
TREES_DEPTH = 3
NUMBER_OF_REGRESSORS = 1

# COLORS
RED = [0, 0, 255]
WHITE = [255, 255, 255]


def plot(image, shape):
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


def warp(points, shape_a, shape_b):
    scale, angle = similarity_transform(shape_b, shape_a)
    warped = np.zeros(points.shape)
    distances = distance(points, shape_a)
    for i in range(len(points)):
        closest_point = np.argmin(distances[i])
        offset = points[i] - shape_a[closest_point]
        offset = rotate(offset / scale, -angle)
        warped[i] = shape_b[closest_point] + offset
    return warped


def process(name):
    real_shape = np.array(dataset[name[:-4]])
    return warp(ref_points, shapes_mean, real_shape)


def get_shapes_mean():
    with open(SHAPES_MEAN_PATH, 'rb') as f:
        return pickle.load(f)


def get_ref_points():
    with open(REF_POINTS_PATH, 'rb') as f:
        return pickle.load(f)

def get_pixel_intensity(data):
    for file_name, points in data.items():
        img = cv2.imread(os.path.join(IMAGE_PATH, file_name), 0)
        intensity_data = []
        for point in points:
            # Fix points that exceed the image limits
            row = point[1].astype(int) % img.shape[0]
            col = point[0].astype(int) % img.shape[1]

            # Appends pixel intensity to corresponding vector
            intensity_data.append(img.item(row, col))

        # Replaces original points information with intensities
        data[file_name] = intensity_data

if __name__ == "__main__":
    print('reading dataset...')
    dataset = read_dataset(DATA_PATH)

    shapes_mean = get_shapes_mean()
    ref_points = get_ref_points()

    print('warping points...')
    p = Pool(4)
    files = os.listdir(IMAGE_PATH)
    data = p.map(process, files)
    data = dict(zip(files, data))

    print('capturing pixel intensity data...')
    get_pixel_intensity(data)
        # cv2.imshow('teste', img)
        # cv2.waitKey(1000)
    print(data)
