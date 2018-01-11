#!/bin/python3.6
import pickle
import os
import numpy as np
import cv2
from time import time
import imutils
from scipy.spatial.distance import cdist as distance
from modules.data_manager import read_dataset
from modules.procrustes import find_theta, rotate

IMAGE_PATH = './img'
DATA_PATH = './data'
MEAN_SHAPE_PATH = 'mean_shape.data'
NUMBER_OF_POINTS = 400
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


if __name__ == "__main__":
    dataset = read_dataset(DATA_PATH)

    with open(MEAN_SHAPE_PATH, 'rb') as f:
        mean_shape = pickle.load(f)

    # The radius is the distance from the origin to the farthest away point
    r = np.max(distance(mean_shape, np.zeros(mean_shape.shape)))
    sorted_points = sort_points(NUMBER_OF_POINTS, [0, 0], r)

    start_time = time()

    group_l = []
    group_r = []

    chosen_points = np.arange(NUMBER_OF_POINTS)
    np.random.shuffle(chosen_points)
    for img_number, file_name in enumerate(os.listdir(IMAGE_PATH)):
        print('Processing image {}: {}'.format(img_number, file_name))
        img = cv2.imread(os.path.join(IMAGE_PATH, file_name), 0)

        real_shape = np.array(dataset[file_name[:-4]])
        estimated_shape = np.copy(mean_shape)

        scale, angle = similarity_transform(real_shape, estimated_shape)

        warped_points = np.zeros(sorted_points.shape)
        for i in range(NUMBER_OF_POINTS):
            closest_point = np.argmin(distance(sorted_points,
                                               estimated_shape)[i])
            offset = sorted_points[i] - estimated_shape[closest_point]
            offset = rotate(offset / scale, -angle)
            warped_points[i] = real_shape[closest_point] + offset

            # Warp overlapping points around image
            if warped_points[i][0] > img.shape[1]:
                warped_points[i][0] = warped_points[i][0] - img.shape[1]
            if warped_points[i][1] > img.shape[0]:
                warped_points[i][1] = warped_points[i][1] - img.shape[0]

        point_u = tuple(warped_points[chosen_points[0]].astype(int))
        point_v = tuple(warped_points[chosen_points[1]].astype(int))

        intensity_u = img.item(point_u[1], point_u[0])
        intensity_v = img.item(point_v[1], point_v[0])
        if (intensity_u - intensity_v) > 50:
            group_l.append(file_name)
            print("left")
        else:
            group_r.append(file_name)
            print("right")

    # for file_name in group_l:
        # draw_shape(img, warped_points)
        # img = cv2.imread(os.path.join(IMAGE_PATH, file_name), 0)
        # img = imutils.resize(img, width=400)
        # cv2.imshow('Frame', img)
        # key = cv2.waitKey(10) & 0xFF
        # if key == 27:
        #     print('ESC key pressed.')
        #     break
print('ellapsed time: {}'.format(time() - start_time))
