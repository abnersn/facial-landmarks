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
parser.add_argument('--regressors', default=10, help='Number of regressors to train.')
parser.add_argument('--trees', default=500, help='Number of trees.')
parser.add_argument('--depth', default=5, help='Trees depth.')
parser.add_argument('--shrinkage', default=0.001, help='Shrinkage factor.')
parser.add_argument('--parameters', default=80, help='Number of parameters to considerer for the PCA.')
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

def first_estimation(item):
    file_name, data = item
    top_left = data['top_left']
    width = data['width']
    height = data['height']

    center = top_left + [width / 2, height / 2]
    scale = 0.3 * width

    data['estimation'] = model.base_shape * scale + center
    return (file_name, data)

log('calculating first estimations')
p = Pool(cpu_count())
dataset = dict(p.map(dataset.items(), ))

def update_data(item):
    file_name, information = item

    estimation = information['estimation']
    sample_points = information['sample_points']
    intensity_data = information['intensity_data']
    tree = information['tree']

    # Retrieve the real shape and image
    real_shape = annotations[file_name[:-4]]
    image = images[file_name]

    # Normalize the coordinates of the estimation in regards to translation
    translation_factor = np.mean(estimation, axis=0)
    estimation_norm = estimation - translation_factor
    
    # Normalize the coordinates of the estimation in regards to scale
    scale_factor = root_mean_square(estimation_norm)
    estimation_norm /= scale_factor

    # Calculate the params that transform the model into the estimation normalized
    params_estimation = model.retrieve_parameters(estimation_norm)

    index = tree.apply(intensity_data)
    delta_params = tree.predictions[index] * SHRINKAGE_FACTOR
    params_estimation += delta_params

    # Calculate new estimation
    new_estimation_norm = model.deform(params_estimation)
    new_estimation = (new_estimation_norm * scale_factor + translation_factor)

    # Calculate new sample points
    new_sample_points = util.warp(sample_points, estimation, new_estimation)

    # Normalize real shape to the current estimation
    real_shape_norm = real_shape - translation_factor
    real_shape_norm /= scale_factor

    # Calculate the parameters that transform the base shape into
    # the normalized versions of both the estimation and the real shape
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    params_estimation = model.retrieve_parameters(new_estimation_norm)

    # Calculate new intensity data
    new_intensity_data = []
    for point in new_sample_points:
        x = max(min(int(point[0]), image.shape[1] - 1), 0)
        y = max(min(int(point[1]), image.shape[0] - 1), 0)
        new_intensity_data.append(image.item(y, x))

    # Update regression data
    new_regression_data = params_real_shape - params_estimation

    return (file_name, {
        'estimation': new_estimation,
        'sample_points': new_sample_points,
        'intensity_data': new_intensity_data,
        'regression_data': new_regression_data
    })


p = Pool(cpu_count())

regressors = []
for r in range(NUMBER_OF_REGRESSORS):
    log('Processing regressor {}...'.format(r + 1))
    labels = data.keys()

    ###############
    ###############
    ###############
    if r == 0:
        data_before = copy.deepcopy(data)
    ###############
    ###############
    ###############

    regressor = []
    for i in range(NUMBER_OF_TREES):
        log('Training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(TREES_DEPTH, labels, data)

        regressor.append(tree)
        for key in data.keys():
            data[key]['tree'] = tree

        # sleep(1)
        log('Updating estimations and sample points...')
        data = dict(p.map(update_data, data.items()))

    regressors.append(regressor)
save('regressors_{}_{}.bin'.format(NUMBER_OF_TREES, NUMBER_OF_REGRESSORS), regressors)

########################################################
log('Calculating error...')
total = len(images.items())
total_error_reduced = 0
average_error_reduction = 0
for file_name, image in images.items():
    real_shape = annotations[file_name[:-4]]
    # BEFORE
    error_current_before = 0
    for i, point in enumerate(data_before[file_name]['estimation']):
        difference = point - real_shape[i]
        square = np.power(difference, 2)
        error_current_before += np.sum(square)
    if DISPLAY:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        util.plot(color, data_before[file_name]['estimation'], util.BLUE)
        util.plot(color, data_before[file_name]['sample_points'], util.GREEN)
        color = resize(color, height=800)
        cv2.imshow('BEFORE', color)

    # AFTER
    error_current_after = 0
    for i, point in enumerate(data[file_name]['estimation']):
        difference = point - real_shape[i]
        square = np.power(difference, 2)
        error_current_after += np.sum(square)
    
    if error_current_after < error_current_before:
        total_error_reduced += 1
        average_error_reduction += 1 - error_current_after / error_current_before
    if DISPLAY:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        util.plot(color, data[file_name]['estimation'], util.BLUE)
        util.plot(color, data[file_name]['sample_points'], util.GREEN)
        color = resize(color, height=800)
        cv2.imshow('AFTER', color)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            break

print('{} of the samples had their errors reduced'.format(
    100 * (total_error_reduced / total)))
print('The average error reduction was of {}'.format(
    100 * (average_error_reduction / total)))
