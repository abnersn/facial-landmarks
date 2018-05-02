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
parser.add_argument('--shrinkage', default=1, help='Shrinkage factor.', type=float)
parser.add_argument('--points', default=600, help='Number of sample points.', type=int)
parser.add_argument('--parameters', default=120, help='Number of parameters to considerer for the PCA.', type=int)
parser.add_argument('-v', '--verbose', action='store_true', help='Whether or not print a detailed output.')
args = parser.parse_args()


def log(message):
    if(args.verbose):
        print(message)

log('reading dataset')
with open(args.dataset_path, 'rb') as f:
    dataset = dill.load(f)

# log('reading model')
# with open('./model.bin', 'rb') as f:
#     model = dill.load(f)

log('calculating PCA model')
model = ShapeModel(args.parameters, calculate_procrustes(dict(
    [(sample['file_name'], sample['annotation']) for sample in dataset]
)))

with open('model.bin', 'wb') as f:
    pickle.dump(model, f)

log('sorting sample points')
radius = 2 * root_mean_square(model.base_shape)
sample_points = util.sort_points(args.points, [0, 0], radius)

with open('sample_points.bin', 'wb') as f:
    pickle.dump(sample_points, f)

def first_estimation(item):
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
    item['sample_points'] = sample_points * scale + pivot

    item['intensity_data'] = []
    for point in item['sample_points']:
        y, x = np.array(point).astype(int)
        try:
            intensity = image.item(x, y)
            item['intensity_data'].append(intensity)
        except IndexError:
            item['intensity_data'].append(0)

    real_shape_norm = item['annotation'] - pivot
    real_shape_norm /= scale
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    item['params_real_shape'] = params_real_shape
    params_estimation = model.retrieve_parameters(model.base_shape)
    item['regression_data'] = params_real_shape - params_estimation
    return item

log('calculating first estimations')
p = Pool(4)
dataset = list(map(first_estimation, dataset))
p.close()
p.join()

# for sample in dataset:
#     image = sample['image']
#     estimation = sample['estimation']
#     annotation = sample['annotation']

#     util.plot(image, annotation, util.BLACK)
#     util.plot(image, estimation, util.WHITE)

#     cv2.imshow('image', image)
#     k = cv2.waitKey(0) & 0xFF
#     if k == 27:
#         sys.exit(0)

def update_data(item):
    # Normalize the estimation
    estimation_norm = ((item['estimation']
                        - item['pivot'])
                        / item['scale'])

    # Displace the parameters according to the prediction
    params_estimation = model.retrieve_parameters(estimation_norm)

    # Calculate the tree prediction
    for tree in item['trees']:
        index = tree.apply(item['intensity_data'])
        prediction = tree.predictions[index] / len(item['trees'])
        params_estimation += prediction * args.shrinkage

    # Update regression data
    new_regression_data = item['params_real_shape'] - params_estimation
    item['regression_data'] = new_regression_data

    # Calculate the new estimation with the displaced parameters
    new_estimation_norm = model.deform(params_estimation)

    # Take the estimation back into position
    new_estimation = (new_estimation_norm
                      * item['scale']
                      + item['pivot'])
    
    # Update data
    item['estimation'] = new_estimation

    # Warp the points 
    new_sample_points = []
    for group in util.warp(sample_points, new_estimation_norm, model.base_shape):
        new_sample_points.append(group * item['scale'] + item['pivot'])
    item['sample_points'] = new_sample_points


    for i, point in enumerate(item['sample_points']):
        y, x = np.array(point).astype(int)
        try:
            intensity = item['image'].item(x, y)
            item['intensity_data'][i] = intensity
        except IndexError:
            item['intensity_data'][i] = 0

    return item

regressors = []
for r in range(args.regressors):
    log('processing regressor {}...'.format(r + 1))

    regressor = []
    for i, sample in enumerate(dataset):
        dataset[i]['trees'] = []
    for i in range(args.trees):
        log('training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(args.depth, dataset)

        regressor.append(tree)

        # Set the new tree for later update
        for i, sample in enumerate(dataset):
            dataset[i]['trees'].append(tree)

    log('updating estimations and sample points...')
    dataset = list(map(update_data, dataset))

    # DEBUG
    sample = dataset[10]
    _image = np.copy(sample['image'])
    _estimation = sample['estimation']
    _annotation = sample['annotation']

    util.plot(_image, _annotation, util.BLACK)
    util.plot(_image, _estimation, util.WHITE)


    cv2.imshow('image', _image)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break

    regressors.append(regressor)

with open('reg.bin', 'wb') as f:
    pickle.dump(regressors, f)

for sample in dataset:
    image = sample['image']
    estimation = sample['estimation']
    annotation = sample['annotation']
    sample_points = np.array(sample['sample_points'])

    util.plot(image, annotation, util.BLACK)
    util.plot(image, estimation, util.WHITE)

    cv2.imshow('image', image)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        sys.exit(0)