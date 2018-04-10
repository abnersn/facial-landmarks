#!/bin/python3.6
import argparse
import pickle, dill
import os, sys
import numpy as np
import datetime
import cv2, dlib
import modules.util as util
from time import sleep
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from multiprocessing import Pool, Process, cpu_count
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

parser = argparse.ArgumentParser(description='This script will train a set of regression trees over a preprocessed dataset.')
parser.add_argument('dataset_path', help='Directory to load the pre processed data from.')
parser.add_argument('--regressors', default=10, help='Number of regressors to train.', type=int)
parser.add_argument('--trees', default=500, help='Number of trees.', type=int)
parser.add_argument('--depth', default=3, help='Trees depth.', type=int)
parser.add_argument('--shrinkage', default=0.1, help='Shrinkage factor.', type=float)
parser.add_argument('--points', default=3, help='Number of sample point per shape point.', type=int)
parser.add_argument('--parameters', default=120, help='Number of parameters to considerer for the PCA.', type=int)
parser.add_argument('-v', '--verbose', action='store_true', help='Whether or not print a detailed output.')
args = parser.parse_args()

def log(message):
    if(args.verbose):
        print(message)

log('reading dataset')
with open(args.dataset_path, 'rb') as f:
    dataset = dill.load(f)

log('calculating PCA model')
model = ShapeModel(args.parameters, calculate_procrustes(dict(
    [(file_name, data['annotations']) for file_name, data in dataset.items()]
)))

with open('model.bin', 'wb') as f:
    pickle.dump(model, f)

log('sorting sample points')
radius = 0.60 * root_mean_square(model.base_shape)
sample_points = np.zeros([len(model.base_shape), args.points, 2])
for i, point in enumerate(model.base_shape):
    sample_points[i] = util.sort_points(args.points, point, radius)

with open('sample_points.bin', 'wb') as f:
    pickle.dump(sample_points, f)

def first_estimation(item):
    file_name, data = item
    image = data['image']
    top_left = data['top_left']
    width = data['width']
    height = data['height']

    pivot = top_left + [width / 2, height / 2]
    scale = 0.3 * width

    data['pivot'] = pivot
    data['scale'] = scale
    data['estimation'] = model.base_shape * scale + pivot
    data['first_estimation'] = model.base_shape * scale + pivot
    data['sample_points'] = []
    for group in sample_points:
        data['sample_points'].append(group * scale + pivot)
    data['intensity_data'] = []

    for group in data['sample_points']:
        intensity_group = []
        for point in group:
            y, x = np.array(point).astype(int)
            try:
                intensity = image.item(x, y)
                intensity_group.append(intensity)
            except IndexError:
                intensity_group.append(-1)
        data['intensity_data'].append(intensity_group)

    real_shape_norm = data['annotations'] - pivot
    real_shape_norm /= scale
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    data['params_real_shape'] = params_real_shape
    params_estimation = model.retrieve_parameters(model.base_shape)
    data['regression_data'] = params_real_shape - params_estimation
    return (file_name, data)

log('calculating first estimations')
p = Pool(4)
dataset = dict(p.map(first_estimation, dataset.items()))

def warp(shape_a, shape_b, groups):
    scale, angle, _ = util.similarity_transform(shape_b, shape_a)
    new_groups = np.zeros(groups.shape)
    for i, group in enumerate(groups):
        for j, point in enumerate(group):
            offset = point - shape_a[i]
            offset = util.rotate(offset / scale, -angle)
            new_groups[i][j] = shape_b[i] - offset
    return new_groups

def update_data(item):
    file_name, data = item

    # Calculate the tree prediction
    index = data['tree'].apply(data['intensity_data'])
    prediction = data['tree'].predictions[index]

    # Normalize the estimation
    estimation_norm = ((data['estimation']
                        - data['pivot'])
                        / data['scale'])

    # Displace the parameters according to the prediction
    params_estimation = model.retrieve_parameters(estimation_norm)
    params_estimation += prediction * args.shrinkage

    # Update regression data
    new_regression_data = data['params_real_shape'] - params_estimation
    data['regression_data'] = new_regression_data

    # Calculate the new estimation with the displaced parameters
    new_estimation_norm = model.deform(params_estimation)

    # Take the estimation back into position
    new_estimation = (new_estimation_norm
                      * data['scale']
                      + data['pivot'])
    
    # Update data
    data['estimation'] = new_estimation

    # Warp the points 
    new_sample_points = []
    for group in warp(new_estimation_norm, model.base_shape, sample_points):
        new_sample_points.append(group * data['scale'] + data['pivot'])
    data['sample_points'] = new_sample_points

    new_intensity_data = []
    for group in new_sample_points:
        intensity_group = []
        for point in group:
            y, x = np.array(point).astype(int)
            try:
                intensity = data['image'].item(x, y)
                intensity_group.append(intensity)
            except IndexError:
                intensity_group.append(-1)
        new_intensity_data.append(intensity_group)
    data['intensity_data'] = new_intensity_data

    return (file_name, data)

p = Pool(4)

regressors = []
for r in range(args.regressors):
    log('processing regressor {}...'.format(r + 1))
    labels = dataset.keys()

    regressor = []
    for i in range(args.trees):
        log('training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(args.depth, labels, dataset, model)

        regressor.append(tree)

        # Set the new tree for later update
        for key in dataset.keys():
            dataset[key]['tree'] = tree

        log('updating estimations and sample points...')
        dataset = dict(p.map(update_data, dataset.items()))

    regressors.append(regressor)

with open('reg.bin', 'wb') as f:
    pickle.dump(regressors, f)

for file_name, data in dataset.items():
    image = data['image']
    estimation = data['estimation']
    first_estimation = data['first_estimation']
    sample_points = np.array(data['sample_points'])
    util.plot(image, estimation)
    util.plot(image, first_estimation, [0, 0, 0])
    # util.plot(image, sample_points.flatten().reshape([3 * len(model.base_shape), 2]))
    cv2.imshow('image', resize(image, height=500))
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        sys.exit(0)