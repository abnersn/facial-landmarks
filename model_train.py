#!/bin/python3.6
import argparse
import pickle, dill
import os, sys
import numpy as np
import datetime
import cv2, dlib
import modules.util as util
from time import sleep, time
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
parser.add_argument('--depth', default=5, help='Trees depth.', type=int)
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
    [(sample['file_name'], sample['annotation']) for sample in dataset]
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
    file_name = item['file_name']
    image = item['image']
    top_left = item['top_left']
    width = item['width']
    height = item['height']

    pivot = top_left + [width / 2, height / 2]
    scale = 0.3 * width

    item['pivot'] = pivot
    item['scale'] = scale
    item['estimation'] = model.base_shape * scale + pivot
    item['first_estimation'] = model.base_shape * scale + pivot
    item['sample_points'] = []
    for group in sample_points:
        item['sample_points'].append(group * scale + pivot)
    item['intensity_data'] = []

    for group in item['sample_points']:
        intensity_group = []
        for point in group:
            y, x = np.array(point).astype(int)
            try:
                intensity = image.item(x, y)
                intensity_group.append(intensity)
            except IndexError:
                intensity_group.append(-1)
        item['intensity_data'].append(intensity_group)

    real_shape_norm = item['annotation'] - pivot
    real_shape_norm /= scale
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    item['params_real_shape'] = params_real_shape
    params_estimation = model.retrieve_parameters(model.base_shape)
    item['regression_data'] = params_real_shape - params_estimation
    return item

log('calculating first estimations')
p = Pool(4)
dataset = p.map(first_estimation, dataset)
p.close()
p.join()

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
    file_name = item['file_name']
    begin = time()

    # Calculate the tree prediction
    calc_prediction = time()
    index = item['tree'].apply(item['intensity_data'])
    prediction = item['tree'].predictions[index]
    calc_prediction = time() - calc_prediction

    # Normalize the estimation
    calc_normalize = time()
    estimation_norm = ((item['estimation']
                        - item['pivot'])
                        / item['scale'])
    calc_normalize = time() - calc_normalize

    # Displace the parameters according to the prediction
    calc_displace = time()
    params_estimation = model.retrieve_parameters(estimation_norm)
    params_estimation += prediction * args.shrinkage
    calc_displace = time() - calc_displace

    # Update regression data
    calc_update = time()
    new_regression_data = item['params_real_shape'] - params_estimation
    item['regression_data'] = new_regression_data
    calc_update = time() - calc_update

    # Calculate the new estimation with the displaced parameters
    calc_new_norm = time()
    new_estimation_norm = model.deform(params_estimation)
    calc_new_norm = time() - calc_new_norm

    # Take the estimation back into position
    new_estimation = (new_estimation_norm
                      * item['scale']
                      + item['pivot'])
    
    # Update data
    item['estimation'] = new_estimation

    # Warp the points 
    calc_warp = time()
    new_sample_points = []
    for group in warp(new_estimation_norm, model.base_shape, sample_points):
        new_sample_points.append(group * item['scale'] + item['pivot'])
    item['sample_points'] = new_sample_points
    calc_warp = time() - calc_warp


    calc_intensity = time()
    new_intensity_data = []
    for group in new_sample_points:
        intensity_group = []
        for point in group:
            y, x = np.array(point).astype(int)
            try:
                intensity = item['image'].item(x, y)
                intensity_group.append(intensity)
            except IndexError:
                intensity_group.append(-1)
        new_intensity_data.append(intensity_group)
    item['intensity_data'] = new_intensity_data
    calc_intensity = time() - calc_intensity

    total = time() - begin

    return data

p = Pool(4)

regressors = []
for r in range(args.regressors):
    log('processing regressor {}...'.format(r + 1))
    labels = [sample['file_name'] for sample in dataset]

    regressor = []
    for i in range(args.trees):
        log('training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(args.depth, labels, dataset, model)

        regressor.append(tree)

        # Set the new tree for later update
        for key in dataset.keys():
            dataset[key]['tree'] = tree

        log('updating estimations and sample points...')
        dataset = p.map(update_data, dataset)

    regressors.append(regressor)

p.close()
p.join()

with open('reg.bin', 'wb') as f:
    pickle.dump(regressors, f)

for sample in dataset:
    image = sample['image']
    estimation = sample['estimation']
    first_estimation = sample['first_estimation']
    sample_points = np.array(sample['sample_points'])
    util.plot(image, estimation)
    util.plot(image, first_estimation, [0, 0, 0])
    cv2.imshow('image', resize(image, height=500))
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        sys.exit(0)