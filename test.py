import numpy as np
import cv2
import imutils
import pickle
import dlib
import os
from scipy.spatial.distance import cdist as distance
from modules.data_manager import read_dataset
from modules.procrustes import find_theta, rotate

# TODO: use cmd line arguments instead of constants
IMAGE_PATH = './img'
DATA_PATH = './data'
MEAN_SHAPE = 'mean_shape.data'
NUMBER_OF_POINTS = 400

# COLORS
RED = [0, 0, 255]
WHITE = [255, 255, 255]

def draw_shape(image, shape):
    for i, point in enumerate(shape):
        draw_point = tuple(np.array(point).astype(int))
        cv2.circle(image, draw_point, 3, WHITE, thickness=-1)

def sort_points(quantity, centroid, radius):
    angles_base = np.full([quantity, 1], 2 * np.pi)
    angles_random = np.random.rand(quantity, 1)
    angles = np.multiply(angles_base, angles_random)
    x_coords = np.multiply(np.cos(angles), np.random.rand(quantity, 1) * radius)
    y_coords = np.multiply(np.sin(angles), np.random.rand(quantity, 1) * radius)
    return np.concatenate((x_coords, y_coords), axis=1) + centroid

def scale(shape_a, shape_b):
    return np.sum(np.multiply(shape_a, shape_b))/np.sum(np.power(shape_b, 2))

detector = dlib.get_frontal_face_detector()
dataset = read_dataset(DATA_PATH)

with open(MEAN_SHAPE, 'rb') as f:
    mean_shape_basis = pickle.load(f)
    # The radius is the distance from the origin to the farthest away point
    radius = np.max(distance(mean_shape_basis, np.zeros(mean_shape_basis.shape)))
    sorted_points = sort_points(NUMBER_OF_POINTS, [0, 0], radius * 2)

for file_name in os.listdir(IMAGE_PATH):
    img = cv2.imread(os.path.join(IMAGE_PATH, file_name), 0)

    mean_shape = np.copy(mean_shape_basis)
    real_shape = np.array(dataset[file_name[:-4]])

    # Calculate the scale, angle of rotation and translation matrix that
    # transforms the mean shape into the real shape

    # Scale
    scale_factor = (np.sum(np.multiply(mean_shape, real_shape))/
        np.sum(np.power(mean_shape, 2)))
    print(scale_factor)
    scaled_shape = mean_shape * scale_factor

    # Rotation angle
    rotation_angle = find_theta(real_shape, scaled_shape)
    rotated_shape = rotate(scaled_shape, rotation_angle)

    # Translation
    translation_matrix = (np.mean(rotated_shape, axis=0) -
        np.mean(real_shape, axis=0))
    translated_shape = rotated_shape - translation_matrix

    new_sorted_points = []
    for i, point in enumerate(sorted_points):
        closest_landmark = np.argmin(distance(sorted_points, mean_shape)[i])
        offset = sorted_points[i] - mean_shape[closest_landmark]
        offset = (rotate(scale_factor * offset, rotation_angle) +
            translation_matrix)
        new_sorted_points.append(sorted_points[i] - offset)

    draw_shape(img, new_sorted_points)

    # draw_shape(img, mean_shape)
    img = imutils.resize(img, width=400)
    cv2.imshow('Frame', img)
    key = cv2.waitKey(1000) & 0xFF
    if key is 27:
        print('ESC key pressed.')
        break
