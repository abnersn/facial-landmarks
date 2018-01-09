"""This modules applies the Procrustes Analysis method to normalize the shapes
of the dataset.

Reference: wikipedia.org/wiki/Procrustes_analysis
"""
__author__ = "Abner S. Nascimento"
__copyright__ = "Copyright 2017, Facial Recognition Project"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "abnersousanascimento@gmail.com"
__status__ = "Development"
import numpy as np

np.set_printoptions(suppress=True)


def translate_mean(matrix):
    """Translates the shape to the center of the coordinates system.

    Args:
        matrix: A numpy matrix that contains the shape points.

    Returns:
        The translated matrix.
    """
    mean = np.mean(matrix, axis=0)
    return matrix - mean


def root_mean_square(matrix):
    """Calculates the root mean squared distance of a set of points

    Args:
        matrix: A numpy matrix that contains the points.

    Returns:
        A float with the root mean squared distance of the given matrix.
    """
    square = np.power(matrix, 2)
    square_mean = np.mean(square)
    return np.sqrt(square_mean)


def scale_rms(matrix):
    """Scales a given shape so that the root mean squared distance of its
    points be equal to 1.

    Args:
        matrix: A numpy matrix that will be scaled.

    Returns:
        The scaled matrix
    """
    rms = root_mean_square(matrix)
    return matrix / rms


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


def square_distance(matrix_a, matrix_b):
    """Calculates the sum of the squared distances between the corresponding
    points of two sets.

    Args:
        matrix_a: A numpy matrix that contains a set of points.
        matrix_b: A numpy matrix that contains a set of points.

    Returns:
        The sum of the squared distances.
    """
    subtraction = matrix_a - matrix_b
    square = np.power(subtraction, 2)
    return np.sum(square)


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


def frange(start, stop, step):
    """A float version of python's range function. Generates a list that starts
    in start and ends at stop, by the given step.

    Args:
        start: First number of the list.
        stop: Last number of the list.
        step: Step between the iterations.

    Yields:
        A list from start to stop, by the given step.
    """
    i = start
    while i < stop:
        yield i
        i += step


def mean_of_shapes(dataset):
    """Calculates the average shape of a dataset.

    Args:
        dataset: A dictionary containing the shapes.
    Returns:
        A set of points that forms the average shape.
    """
    sum_of_samples = np.zeros(next(iter(dataset.values())).shape)
    for points in dataset.values():
        sample = np.array(points)
        sum_of_samples += sample
    return sum_of_samples / len(dataset)


def calculate_procrustes(dataset):
    """Performs Procrustes analysis on a given dataset.

    Args: A dataset as a python dictionary in which keys are image filenames
    and values are its corresponding vector of tuples that represents annotated
    shapes.

    Returns: A new dictionary with image filenames as keys and a Procrustes
    transformed numpy array.
    """
    new_dataset = {}
    reference_key = next(iter(dataset))
    new_dataset[reference_key] = np.array(dataset[reference_key])
    new_dataset[reference_key] = translate_mean(new_dataset[reference_key])
    new_dataset[reference_key] = scale_rms(new_dataset[reference_key])

    for image_file, points_list in dataset.items():
        current_mean = mean_of_shapes(new_dataset)
        shape = np.array(points_list)
        translated_shape = translate_mean(shape)
        scaled_shape = scale_rms(translated_shape)
        found_theta = find_theta(current_mean, shape)
        rotated_shape = rotate(scaled_shape, found_theta)
        new_dataset[image_file] = rotated_shape

    return new_dataset
