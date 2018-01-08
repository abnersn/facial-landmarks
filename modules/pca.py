"""This module implements the Principal Component Analysis, or PCA, for short. The
PCA is a tool commonly used to solve problems involving high dimensional data, due
to its dimensional reduction properties.
"""
__author__ = "Abner S. Nascimento"
__copyright__ = "Copyright 2017, Facial Recognition Project"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "abnersousanascimento@gmail.com"
__status__ = "Development"
import numpy as np

def mean_of_faces(dataset):
    """Calculates the average face of a dataset of shapes.

    Args:
        dataset: A dictionary containing the shapes of the faces.
    Returns:
        A set of points that forms the shape of the average face
    """
    sum_of_samples = np.zeros([194, 2])
    for points in dataset.values():
        sample = np.array(points)
        sum_of_samples += sample
    return sum_of_samples / len(dataset)

def covariance(dataset):
    """Calculates the covariance matrix of the data.

    Args:
        dataset: A dictionary containing the shapes to which the covariance calculus
                 will be performed.
    Returns:
        The covariance matrix.
    """
    mean = mean_of_faces(dataset)
    sum_of_samples = np.zeros([194, 194])
    for points in dataset.values():
        sample = np.array(points)
        sum_of_samples += np.dot((sample - mean), np.transpose(sample-mean))
    return sum_of_samples / (len(dataset) - 1)

def perform_pca(dataset, number_of_params):
    """Performs the Principal Component Analysis over a high dimensional dataset.

    Args:
        dataset: A dictionary containing the data to which the PCA shall be applied.
        number_of_params: Number of parameters (eigenvalues and eigenvectors) desired
                          for the output.
    Returns:
        The eigenvalues and eigenvectors of the covariance matrix that corresponds
        to the result of the PCA.
    """
    covariance_matrix = covariance(dataset)

    e_values, e_vectors = np.linalg.eig(covariance_matrix)
    e_values = e_values[0:number_of_params]
    e_vectors = e_vectors[:, 0:number_of_params]

    return (e_values, e_vectors)
