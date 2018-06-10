#!/bin/python3.6
import argparse
import pickle, dill
import os, sys
import numpy as np
import cv2
import modules.util as util
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from modules.procrustes import calculate_procrustes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

parser = argparse.ArgumentParser(description='This script will train a set of regression trees over a preprocessed dataset.')
parser.add_argument('dataset_path', help='Directory to load the pre processed data from.')
parser.add_argument('-r', '--regressors', default=15, help='Number of regressors to train.', type=int)
parser.add_argument('-t', '--trees', default=30, help='Number of trees.', type=int)
parser.add_argument('-d', '--depth', default=5, help='Trees depth.', type=int)
parser.add_argument('-q', '--points', default=600, help='Number of sample points.', type=int)
parser.add_argument('-p', '--parameters', default=70, help='Number of parameters to considerer for the PCA.', type=int)
parser.add_argument('--silent', action='store_true', help='Turn on silent mode, output will not be printed.')
args = parser.parse_args()


def log(message):
    if not args.silent:
        print(message)

log('reading dataset')
with open(args.dataset_path, 'rb') as f:
    dataset = dill.load(f)

log('calculating PCA model')
model = ShapeModel(args.parameters, calculate_procrustes(dict(
    [(sample['file_name'], sample['annotation']) for sample in dataset]
)))

with open('model.data', 'wb') as f:
    pickle.dump(model, f)

# with open('model.data', 'rb') as f:
#     model = pickle.load(f)

log('sorting sample points')
RADIUS = 2 * root_mean_square(model.base_shape)
sample_points = util.sort_points(args.points, [0, 0], RADIUS)

with open('sample_points.data', 'wb') as f:
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
dataset = list(map(first_estimation, dataset))

# DEBUG /start
# sample = dataset[45]
# _image = np.copy(sample['image'])
# _estimation = sample['estimation']
# _annotation = sample['annotation']

# # util.plot(_image, _annotation, util.BLACK)
# _image = np.zeros(_image.shape)
# util.plot(_image, sample['sample_points'], util.WHITE)


# cv2.imshow('image', _image)
# k = cv2.waitKey(0) & 0xFF
# sys.exit()
# DEGUG /end

regressors = []
current_regressor = 0

def update_data(item):
    # Normalize the estimation
    estimation_norm = ((item['estimation']
                        - item['pivot'])
                        / item['scale'])

    # Displace the parameters according to the prediction
    params_estimation = model.retrieve_parameters(estimation_norm)

    # Calculate the tree prediction
    for tree in regressors[current_regressor]:
        index = tree.apply(item['intensity_data'])
        prediction = tree.predictions[index] / len(regressors[current_regressor])
        params_estimation += prediction

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
        # If the sample point is outside of the image borders
        except IndexError:
            item['intensity_data'][i] = 0

    return item

for r in range(args.regressors):
    log('processing regressor {}...'.format(r + 1))

    current_regressor = r

    regressor = []
    for i in range(args.trees):
        log('training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(args.depth, dataset)

        regressor.append(tree)
    regressors.append(regressor)

    log('updating estimations and sample points...')
    dataset = list(map(update_data, dataset))

    # DEBUG /start
    sample = dataset[4]
    _image = np.copy(sample['image'])
    _estimation = sample['estimation']
    _annotation = sample['annotation']

    util.plot(_image, _annotation, util.BLACK)
    util.plot(_image, _estimation, util.WHITE)


    cv2.imshow('image', _image)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break
    # DEGUG /end

with open('reg.data', 'wb') as f:
    pickle.dump(regressors, f)