#!/bin/python3.6
import argparse
import pickle
import dill
import os
import sys
import numpy as np
import cv2
import modules.util as util
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from modules.procrustes import calculate_procrustes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

parser = argparse.ArgumentParser(
    description='This script will train a set of regression trees over a preprocessed dataset.')
parser.add_argument(
    'dataset_path', help='Directory to load the pre processed data from.')
parser.add_argument('-r', '--regressors', default=10,
                    help='Number of regressors to train.', type=int)
parser.add_argument('-t', '--trees', default=500,
                    help='Number of trees.', type=int)
parser.add_argument('-d', '--depth', default=3, help='Trees depth.', type=int)
parser.add_argument('-q', '--points', default=400,
                    help='Number of sample points.', type=int)
parser.add_argument('-p', '--parameters', default=76,
                    help='Number of parameters to considerer for the PCA.', type=int)
parser.add_argument('-o', '--output', default='model.data',
                    help='Output filename.', type=str)
parser.add_argument('--silent', action='store_true',
                    help='Turn on silent mode, output will not be printed.')
parser.add_argument('--safe', action='store_true',
                    help='Turn on safe mode, regressors will be saved after each iteration of the training process.')
parser.add_argument('--range', type=str, default='0-190')
args = parser.parse_args()

model = {}


def log(message):
    if not args.silent:
        print(message)


log('reading dataset')
with open(args.dataset_path, 'rb') as f:
    dataset = dill.load(f)

[start, end] = args.range.split('-')
del dataset[int(start):int(end)]

log('Processing {} images'.format(len(dataset)))

log('calculating PCA model')
pca_model = ShapeModel(args.parameters, calculate_procrustes(dict(
    [(sample['file_name'], sample['annotation']) for sample in dataset]
)))

model['pca_model'] = pca_model

log('sorting sample points')
RADIUS = 2 * root_mean_square(pca_model.base_shape)
sample_points = util.sort_points(args.points, [0, 0], RADIUS)

model['sample_points'] = sample_points


def first_estimation(item):
    image = item['image']
    top_left = item['top_left']
    width = item['width']
    height = item['height']

    pivot = top_left + [width / 2, height / 2]
    scale = 0.4 * width

    item['pivot'] = pivot
    item['scale'] = scale
    item['estimation'] = pca_model.base_shape * scale + pivot
    item['first_estimation'] = pca_model.base_shape * scale + pivot
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
    params_real_shape = pca_model.retrieve_parameters(real_shape_norm)
    item['params_real_shape'] = params_real_shape
    params_estimation = pca_model.retrieve_parameters(pca_model.base_shape)
    item['regression_data'] = params_real_shape - params_estimation
    return item


log('calculating first estimations')
dataset = list(map(first_estimation, dataset))

regressors = []
current_regressor = []


def update_estimation(item):
    # Normalize the estimation
    estimation_norm = ((item['estimation']
                        - item['pivot'])
                       / item['scale'])

    # Displace the parameters according to the prediction
    params_estimation = pca_model.retrieve_parameters(estimation_norm)

    # Calculate the tree prediction
    # for tree in current_regressor:
    tree = current_regressor[-1]
    index = tree.apply(item['intensity_data'])
    prediction = tree.predictions[index] * 0.1
    params_estimation += prediction

    # Update regression data
    new_regression_data = item['params_real_shape'] - params_estimation
    item['regression_data'] = new_regression_data

    # Calculate the new estimation with the displaced parameters
    new_estimation_norm = pca_model.deform(params_estimation)

    # Take the estimation back into position
    new_estimation = (new_estimation_norm
                      * item['scale']
                      + item['pivot'])

    # Update data
    if len(current_regressor) == 1:
        item['previous_estimation'] = np.copy(item['estimation'])
    item['estimation'] = new_estimation

    return item


def update_warping(item):
    # Warp the points
    item['sample_points'] = util.warp(
        item['sample_points'],
        item['previous_estimation'],
        item['estimation']
    )

    for i, point in enumerate(item['sample_points']):
        y, x = np.array(point).astype(int)
        try:
            intensity = item['image'].item(x, y)
            item['intensity_data'][i] = intensity
        # If the sample point is outside of the image borders
        except IndexError:
            item['intensity_data'][i] = 0
    return item


debug_sample = 4

for r in range(args.regressors):
    log('processing regressor {}...'.format(r + 1))

    current_regressor = []

    for i in range(args.trees):
        log('training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(args.depth, dataset)
        current_regressor.append(tree)

        log('updating estimations')
        dataset = list(map(update_estimation, dataset))
        # DEBUG /start
        sample = dataset[debug_sample]
        _image = cv2.cvtColor(np.copy(sample['image']), cv2.COLOR_GRAY2BGR)
        _estimation = sample['estimation']
        _sample_points = sample['sample_points']
        _annotation = sample['annotation']

        # print(sample['regression_data'])

        util.plot(_image, _annotation, util.BLACK)
        # util.plot(_image, _sample_points, util.BLUE)
        util.plot(_image, _estimation, util.WHITE)

        #cv2.imshow('image', _image)
        #k = cv2.waitKey(100) & 0xFF
        #if k == 27:
        #    sys.exit()
        #elif k == 110:
        #    debug_sample -= 1
        #    log('sample {}'.format(debug_sample))
        #elif k == 109:
        #    debug_sample += 1
        #    log('sample {}'.format(debug_sample))
        #debug_sample = min(max(debug_sample, 0), 1999)
        # DEGUG /end
    regressors.append(current_regressor)

    log('updating sample points...')
    dataset = list(map(update_warping, dataset))

    if args.safe:
        model['regressors'] = regressors

if not args.safe:
    model['regressors'] = regressors

if not args.safe:
    with open(args.output, 'wb') as f:
        pickle.dump(model, f)
