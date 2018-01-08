"""This module calculates the profiles of the points that form a shape.
"""
__author__ = "Abner S. Nascimento"
__copyright__ = "Copyright 2017, Facial Recognition Project"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "abnersousanascimento@gmail.com"
__status__ = "Development"
import numpy as np
from bresenham import bresenham

def normal_vector(point_a, point_b):
    """ Calculates the normal vector between two points.
    
    Args:
        point_a: First point
        point_b: Second point
    Returns:
        A normalized vector that points in the direction of
        the normal between point a and point b.
    """
    vector = np.subtract(point_a, point_b)
    normal = (vector[1], -vector[0])
    return normal / np.linalg.norm(normal)

def calculate_normals(shape):
    """ Calculates the normals vectors of every point in a shape.
    
    Args:
        shape: Shape from which the normal points will be
        calculated.
    Returns:
        A set of normalized vectors that point in the direction
        of the corresponding normal to the shape edge.
    """
    subshapes = {
        'jawline': (0, 41), 
        'nose': (41, 58), 
        'mouth_external': (58, 86), 
        'mouth_internal': (86, 114),
        'left_eye': (114, 134),
        'right_eye': (134, 154),
        'left_eyebrow': (154, 174),
        'right_eyebrow': (174, 194)
    }
    normals = np.zeros([len(shape), 2])
    for key in subshapes: # Loops through the subshapes
        points_range = range(subshapes[key][0], subshapes[key][1])
        length = len(list(points_range))
        for i in range(length):
            b = subshapes[key][0] + i
            a = subshapes[key][0] + ((i - 1) % length)
            c = subshapes[key][0] + ((i + 1) % length)
            normals[b] = normal_vector(shape[a], shape[c])
    return normals

def calculate_profiles(shape, profile_size):
    """ Extracts a set of points that define a stright line
    in the direction of the normals of a shape, called profile.
    The profile of a point is centered at its coordinates.
    
    Args:
        shape: Shape from which the normal lines will be calculated.
        profile_size: Number of points on each side of the line.
    Returns:
        A set of normalized vectors that point in the direction
        of the corresponding normal to the shape edge.
    """
    normals = calculate_normals(shape)
    profiles = []
    for i, point in enumerate(shape):
        normal_a = np.add(point, -profile_size * normals[i]).astype(int)
        normal_b = np.add(point, profile_size * normals[i]).astype(int)

        profiles.append(list(bresenham(normal_a[0], normal_a[1], normal_b[0], normal_b[1])))
    return profiles