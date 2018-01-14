#!/bin/python3.6
from multiprocessing import Pool
from time import time
import pickle
import os, sys
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
SHRINKAGE_FACTOR = 0.01

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

    return (scale_factor, rotation_angle, translation_matrix)


def warp(points, shape_a, shape_b):
    scale, angle, _ = similarity_transform(shape_b, shape_a)
    warped = np.zeros(points.shape)
    distances = distance(points, shape_a)
    for i in range(len(points)):
        closest_point = np.argmin(distances[i])
        offset = points[i] - shape_a[closest_point]
        offset = rotate(offset / scale, -angle)
        warped[i] = shape_b[closest_point] + offset
    return warped


def get_shapes_mean():
    with open(SHAPES_MEAN_PATH, 'rb') as f:
        return pickle.load(f)


def get_ref_points():
    with open(REF_POINTS_PATH, 'rb') as f:
        return pickle.load(f)

def get_pixel_intensity(shape_data):
    for file_name, points in shape_data.items():
        img = cv2.imread(os.path.join(IMAGE_PATH, file_name), 0)
        intensity_data = []
        for point in points:
            # Fix points that exceed the image limits
            row = point[1].astype(int) % img.shape[0]
            col = point[0].astype(int) % img.shape[1]

            # Appends pixel intensity to corresponding vector
            intensity_data.append(img.item(row, col))

        # Replaces original points information with intensities
        shape_data[file_name] = intensity_data
    return shape_data


def process(name):
    real_shape = np.array(dataset[name[:-4]])
    return warp(ref_points, shapes_mean, real_shape)


def split_node(files, split, intensity_data):
    left = []
    right = []
    for file_name in files:
        intensity_u = intensity_data[file_name][split[0]]
        intensity_v = intensity_data[file_name][split[1]]
        if intensity_u - intensity_v > split[2]:
            left.append(file_name)
        else:
            right.append(file_name)
    return (left, right)

def grow_tree(files, splits, intensity_data):
    nodes_queue = [files]
    levels_queue = [0]
    for i in range(pow(2, len(splits)) - 1):
        node = nodes_queue.pop(0)
        level = levels_queue.pop(0)
        print('splitting node {} by split criteria {}...'.format(i, level))
        left, right = split_node(node, splits[level], intensity_data)
        nodes_queue.append(left)
        levels_queue.append(level + 1)

        nodes_queue.append(right)
        levels_queue.append(level + 1)
    return nodes_queue


if __name__ == "__main__":
    print('reading dataset...')
    dataset = read_dataset(DATA_PATH)
    files = os.listdir(IMAGE_PATH)

    shapes_mean = get_shapes_mean()
    ref_points = get_ref_points()

    print('warping points...')
    p = Pool(4)
    data = p.map(process, files)
    data = dict(zip(files, data))

    print('capturing pixel intensity data...')
    data = get_pixel_intensity(data)

    # for k in range(NUMBER_OF_TREES):
    print('growing tree...')
    tree_splits = []
    for f in range(TREES_DEPTH):
        sort_aux = np.arange(len(ref_points))
        np.random.shuffle(sort_aux)
        u = sort_aux[0]
        v = sort_aux[1]
        threshold = np.random.randint(255)
        tree_splits.append((u, v, threshold))
    tree = grow_tree(files, tree_splits, data)
    for leaf in tree:
        delta_landmarks = np.zeros(shapes_mean.shape)
        for file_name in leaf:
            real_shape = np.array(dataset[file_name[:-4]])
            s, _, t = similarity_transform(real_shape, shapes_mean)
            estimation = (shapes_mean / s) + t
            delta_landmarks += real_shape - estimation
            # img = cv2.imread(os.path.join(IMAGE_PATH, file_name), 0)
            # plot(img, estimation)
            # resized = resize(img, width=400)
            # cv2.imshow('image', resized)
            # k = cv2.waitKey(1000) & 0xFF
            # if k == 27:
            #     sys.exit(1)
        delta_landmarks = SHRINKAGE_FACTOR * (delta_landmarks / len(leaf))
        print(delta_landmarks)
    print(len(tree))
