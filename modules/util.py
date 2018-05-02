'''
A collection of helper methods for the facial landmarks detector
implementation.
'''

import os
import numpy as np
import cv2

# COLORS
BLUE = [255, 0, 0]
GREEN = [0, 255, 0]
RED = [0, 0, 255]
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]

def read_annotations(annotations_path):
    '''Reads annotations files from dataset.

    Args:
        annotations_path: Path to where the annotations are stored. Defaults
        to current path.

    Returns:
        Dictionary with image file names as keys and a numpy array of points as
        value for each image.
    '''
    if not annotations_path:
        annotations_path = '.'
    annotation_files = os.listdir(annotations_path)
    images_dictionary = {}

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotations_path, annotation_file)
        with open(annotation_file_path, 'r') as csv_file:
            image_file_name = csv_file.readline().rstrip()
            images_dictionary[image_file_name] = []
            for line in csv_file:
                [point_x, point_y] = line.split(' , ')
                point = (float(point_x), float(point_y))
                images_dictionary[image_file_name].append(point)
            images_dictionary[image_file_name] = np.array(
                images_dictionary[image_file_name]
            )
    return images_dictionary


def read_images(images_path, color=False):
    images = {}
    for file_name in os.listdir(images_path):
        img = cv2.imread(os.path.join(images_path, file_name), int(color))
        images[file_name] = img
    return images

def plot(image, shape, color=WHITE):
    size = int(image.shape[0] * 0.005)
    for point in shape:
        draw_point = tuple(np.array(point).astype(int))
        cv2.circle(image, draw_point, size, color, thickness=-1)

def sort_points(number, center, radius):
    angles_base = np.full([number, 1], 2 * np.pi)
    angles_random = np.random.rand(number, 1)
    angles = np.multiply(angles_base, angles_random)
    x_coords = (np.multiply(np.cos(angles),
                            np.random.rand(number, 1) * radius))
    y_coords = (np.multiply(np.sin(angles),
                            np.random.rand(number, 1) * radius))
    return np.concatenate((x_coords, y_coords), axis=1) + center


def similarity_transform(shape_a, shape_b):
    translation_matrix = (np.mean(shape_a, axis=0) - np.mean(shape_b, axis=0))
    shape_a = shape_a - translation_matrix

    product = np.multiply(shape_a, shape_b)
    scale_factor = (np.sum(product) / np.sum(np.power(shape_a, 2)))
    scaled_shape = shape_a * scale_factor

    rotation_angle = find_theta(shape_b, scaled_shape)

    return (scale_factor, rotation_angle, translation_matrix)


def rotate(matrix, theta):
    """Rotates a shape by a given angle in radians along the coordinates
    system's center.

    Args:
        matrix: A numpy matrix that will be rotated.
        theta: The angle (in radians) of the rotation

    Returns:
        The rotated matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta, cos_theta]])
    result = np.dot(rotation_matrix, np.transpose(matrix))
    return np.transpose(result)


def find_theta(matrix_a, matrix_b):
    """Calculates the angle by which matrix_b should be rotated in order to
    minimize its squared distance in relation to matrix_a. This is the
    analytical solution for the rotation step of the Procrustes analysis.

    Args:
        matrix_a: A numpy matrix that contains a set of points.
        matrix_b: A numpy matrix that contains a set of points.

    Returns:
        The optimal angle in radians.
    """
    x_values_a = matrix_a[:, 0]
    x_values_b = matrix_b[:, 0]

    y_values_a = matrix_a[:, 1]
    y_values_b = matrix_b[:, 1]

    numerator = np.sum(y_values_a * x_values_b - x_values_a * y_values_b)
    denominator = np.sum(x_values_a * x_values_b + y_values_a * y_values_b)

    theta = np.arctan(numerator / denominator)
    return theta


def closest_point(point, shape):
    distances = np.sum((shape - point)**2, axis=1)
    return np.argmin(distances)


def warp(sample_points, estimation, annotation):
    scale, angle, _ = similarity_transform(estimation, annotation)
    warped = np.zeros(sample_points.shape)
    for i, point in enumerate(sample_points):
        closest = closest_point(point, estimation)
        offset = point - estimation[closest]
        offset = rotate(offset * scale, angle)
        warped[i] = annotation[closest] + offset
    return warped