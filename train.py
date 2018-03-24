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
parser.add_argument('--regressors', default=3, help='Number of regressors to train.', type=int)
parser.add_argument('--trees', default=3, help='Number of trees.', type=int)
parser.add_argument('--depth', default=4, help='Trees depth.', type=int)
parser.add_argument('--shrinkage', default=0.001, help='Shrinkage factor.', type=float)
parser.add_argument('--points', default=3, help='Number of sample point per shape point.', type=int)
parser.add_argument('--parameters', default=80, help='Number of parameters to considerer for the PCA.', type=int)
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

log('sorting sample points')
radius = 0.60 * root_mean_square(model.base_shape)
sample_points = np.zeros([len(model.base_shape), args.points, 2])
for i, point in enumerate(model.base_shape):
    sample_points[i] = util.sort_points(args.points, point, radius)

def first_estimation(item):
    file_name, data = item
    image = data['image']
    top_left = data['top_left']
    width = data['width']
    height = data['height']

    center = top_left + [width / 2, height / 2]
    scale = 0.3 * width

    data['estimation'] = model.base_shape * scale + center
    data['sample_points'] = []
    for group in sample_points:
        data['sample_points'].append(group * scale + center)
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

    real_shape_norm = data['annotations'] - center
    real_shape_norm /= root_mean_square(data['estimation'])
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    params_estimation = model.retrieve_parameters(model.base_shape)
    data['regression_data'] = data['annotations'] - data['estimation']
    return (file_name, data)

log('calculating first estimations')
p = Pool(cpu_count())
dataset = dict(p.map(first_estimation, dataset.items()))

def warp2(shape_a, shape_b, groups):
    diff = shape_a - shape_b
    return np.array([group + diff[i] for i, group in enumerate(groups)])

def update_data(item):
    file_name, data = item

    estimation = data['estimation']
    intensity_data = data['intensity_data']
    real_shape = data['annotations']
    image = data['image']
    tree = data['tree']

    # Normalize the coordinates of the estimation in regards to translation
    translation_factor = np.mean(estimation, axis=0)
    estimation_norm = estimation - translation_factor
    
    # Normalize the coordinates of the estimation in regards to scale
    scale_factor = root_mean_square(estimation_norm)
    estimation_norm /= scale_factor

    # Calculate the params that transform the model into the estimation normalized
    params_estimation = model.retrieve_parameters(estimation_norm)

    index = tree.apply(intensity_data)
    delta_params = tree.predictions[index] * args.shrinkage
    params_estimation += delta_params

    # Calculate new estimation
    new_estimation_norm = model.deform(params_estimation)
    new_estimation = (new_estimation_norm * scale_factor + translation_factor)

    # Calculate new sample points
    new_sample_points = []
    for group in warp2(new_estimation_norm, model.base_shape, sample_points):
        new_sample_points.append(group * scale_factor + translation_factor)

    new_intensity_data = []
    for group in new_sample_points:
        intensity_group = []
        for point in group:
            y, x = np.array(point).astype(int)
            try:
                intensity = image.item(x, y)
                intensity_group.append(intensity)
            except IndexError:
                intensity_group.append(-1)
        new_intensity_data.append(intensity_group)
    
    # Normalize real shape to the current estimation
    real_shape_norm = real_shape - translation_factor
    real_shape_norm /= scale_factor

    # Calculate the parameters that transform the base shape into
    # the normalized versions of both the estimation and the real shape
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    params_estimation = model.retrieve_parameters(new_estimation_norm)

    # Update regression data
    new_regression_data = params_real_shape - params_estimation

    return (file_name, {
        'image': image,
        'tree': tree,
        'annotations': real_shape,
        'estimation': new_estimation,
        'sample_points': new_sample_points,
        'intensity_data': new_intensity_data,
        'regression_data': new_regression_data
    })

p = Pool(cpu_count())

regressors = []
for r in range(args.regressors):
    log('processing regressor {}...'.format(r + 1))
    labels = dataset.keys()

    regressor = []
    for i in range(args.trees):
        log('training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(args.depth, labels, dataset)

        regressor.append(tree)
        for key in dataset.keys():
            dataset[key]['tree'] = tree

        log('updating estimations and sample points...')
        dataset = map(update_data, dataset.items())
        dataset = dict(dataset)

    regressors.append(regressor)

for file_name, data in dataset.items():
    image = data['image']
    estimation = data['estimation']
    sample_points = data['sample_points']
    sample_points = np.array(sample_points).flatten().reshape([args.points * len(model.base_shape), 2])
    intensity_data = data['intensity_data']
    size = int(image.shape[0] * 0.005)
    for i, point in enumerate(sample_points):
        intensity = intensity_data[i]
        if intensity != -1:
            draw_point = tuple(np.array(point).astype(int))
            cv2.circle(image, draw_point, size, [intensity, intensity, intensity], thickness=2)
        
    util.plot(image, estimation)
    cv2.imshow('image', resize(image, height=500))
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        sys.exit(0)