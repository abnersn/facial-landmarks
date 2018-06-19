#!/usr/bin/python
import pickle
import numpy as np
import cv2
from modules.util import plot
from modules.face_model import ShapeModel
from modules.procrustes import root_mean_square, calculate_procrustes, find_theta, rotate
from imutils import resize

with open('datasets/training_300', 'rb') as f:
    dataset = pickle.load(f)

NUMBER_OF_PARAMS = 12
FAULTY_PERCENTAGE = 75

model = ShapeModel(NUMBER_OF_PARAMS, calculate_procrustes(dict(
    [(sample['file_name'], sample['annotation']) for sample in dataset]
)))

def correct_pca(faulty_shape):    
    faulty_points = []
    for i, point in enumerate(faulty_shape):
        if np.isnan(point).all():
            faulty_points.append(i)

    # Remove faulty points from base and shape
    faulty_base = np.delete(model.base_shape, faulty_points, axis=0)
    faulty_normalized = np.delete(faulty_shape, faulty_points, axis=0)
    
    # Normalize
    pivot = np.mean(faulty_normalized, axis=0)
    faulty_normalized -= pivot
    scale = root_mean_square(faulty_normalized)
    faulty_normalized /= scale
    theta = find_theta(faulty_base, faulty_normalized)
    faulty_normalized = rotate(faulty_normalized, theta)
    
    # Calculate PCA params
    params = model.retrieve_parameters_faulty(faulty_normalized, faulty_points)
    corrected_normalized = model.deform(params)

    # Scale, rotate and translate back into position
    corrected = corrected_normalized * scale
    corrected = rotate(corrected, -theta)
    corrected += pivot

    return corrected

def correct_interpolate(faulty_shape):
    faulty_groups = [[]]
    for i, point in enumerate(faulty_shape):
        if np.isnan(point).all():
            last_group = faulty_groups[-1]
            if len(last_group) == 0 or last_group[-1] + 1 == i:
                faulty_groups[-1].append(i)
            else:

                faulty_groups.append([i])
    for group in faulty_groups:
        first = min(group)
        if first == 0:
            first = len(faulty_shape) - 1
        else:
            first -= 1
        last = max(group)
        if last == len(faulty_shape) - 1:
            last = 0
        else:
            last += 1
        offset = ((faulty_shape[last] - faulty_shape[first])
                   / (len(group) + 1))
        for i, point_index in enumerate(group):
            faulty_shape[point_index] = (faulty_shape[first]
                                         + offset * (i + 1))
    return faulty_shape
    
sample = dataset[21]

amount_to_remove = int(np.round(194 * FAULTY_PERCENTAGE / 100))
extremal_points = [0, 40, 41, 57, 58, 113, 114, 133, 134, 153, 154, 173, 174, 193]
removal_candidates = [i for i in range(0, 194) if i not in extremal_points]
faulty_points = np.random.choice(removal_candidates, amount_to_remove)

faulty_real_shape = np.copy(sample['annotation'])
faulty_real_shape[faulty_points] = np.array([np.nan, np.nan])

corrected_interpolate = correct_interpolate(faulty_real_shape)
corrected_pca = correct_pca(faulty_real_shape)
img_1 = np.copy(sample['image'])

plot(img_1, corrected_pca)
plot(sample['image'], corrected_interpolate)
cv2.imshow('interpolated', resize(sample['image'], height=750))
cv2.imshow('pca', resize(img_1, height=750))
k = cv2.waitKey(0) & 0xFF
if k == 27:
    pass