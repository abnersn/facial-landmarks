#!/usr/bin/python
import argparse
import pickle, dill
import pickle
import numpy as np
import cv2
from modules.util import plot
from modules.face_model import ShapeModel
from modules.procrustes import root_mean_square, calculate_procrustes, find_theta, rotate
from imutils import resize

parser = argparse.ArgumentParser(description='Build a faulty dataset and recover the missing points using PCA and linear interpolation.')
parser.add_argument('model_path', help='Directory of the complete samples to train the pca model.')
parser.add_argument('dataset_path', help='Directory to load the pre processed data from.')
parser.add_argument('-p', '--params', default=20, help='Number of parameters.', type=int)
parser.add_argument('-c', '--percentage', default=80, help='Percentage of points to eliminate.', type=int)
args = parser.parse_args()

with open(args.model_path, 'rb') as f:
    model_dataset = pickle.load(f)

with open(args.dataset_path, 'rb') as f:
    dataset = pickle.load(f)

complete_samples = [sample['file_name'] for sample in model_dataset]
extremity_points = [0, 40, 41, 57, 58, 113, 114, 133, 134, 153, 154, 173, 174, 193]
removal_candidates = [i for i in range(0, 194) if i not in extremity_points]

model = ShapeModel(args.params, calculate_procrustes(dict(
    [(sample['file_name'], sample['annotation']) for sample in model_dataset]
)))

pca_corrected_dataset = []
interpolation_corrected_dataset = []

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

# Remove points from dataset
for sample in dataset:
    pca_corrected = {
        'file_name': sample['file_name'],
        'image': np.copy(sample['image']),
        'annotation': np.copy(sample['annotation']),
        'top_left': np.copy(sample['top_left']),
        'width': sample['width'],
        'height': sample['height']
    }

    interpolation_corrected = {
        'file_name': sample['file_name'],
        'image': np.copy(sample['image']),
        'annotation': np.copy(sample['annotation']),
        'top_left': np.copy(sample['top_left']),
        'width': sample['width'],
        'height': sample['height']
    }

    if sample['file_name'] not in complete_samples:
        total_points = len(sample['annotation'])
        amount_to_remove = int(np.round(total_points * args.percentage / 100))
        faulty_points = np.random.choice(removal_candidates, amount_to_remove)

        faulty_real_shape = np.copy(sample['annotation'])
        faulty_real_shape[faulty_points] = np.array([np.nan, np.nan])

        interpolation_corrected['annotation'] = correct_interpolate(faulty_real_shape)
        pca_corrected['annotation'] = correct_pca(faulty_real_shape)
    else:
        interpolation_corrected['annotation'] = np.copy(sample['annotation'])
        pca_corrected['annotation'] = np.copy(sample['annotation'])

    interpolation_corrected_dataset.append(interpolation_corrected)
    pca_corrected_dataset.append(pca_corrected)

print('Saving pca corrected dataset')
with open(args.dataset_path.replace('datasets', 'faulty_datasets_pca'), 'wb') as f:
    dill.dump(pca_corrected_dataset, f)

print('Saving interpolation corrected dataset')
with open(args.dataset_path.replace('datasets', 'faulty_datasets_interpolation'), 'wb') as f:
    dill.dump(interpolation_corrected_dataset, f)

for i, sample in enumerate(pca_corrected_dataset):
    pca_corrected_image = np.copy(sample['image'])
    interpolation_corrected_image = np.copy(sample['image'])

    plot(pca_corrected_image, sample['annotation'])
    plot(interpolation_corrected_image, interpolation_corrected_dataset[i]['annotation'])

    cv2.imshow('PCA Corrected', pca_corrected_image)
    cv2.imshow('Interpolation Corrected', interpolation_corrected_image)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        break