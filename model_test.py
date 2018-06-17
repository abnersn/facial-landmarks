#!/bin/python3.6
import argparse
import pickle, dill
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square

parser = argparse.ArgumentParser(description='This script will train a set of regression trees over a preprocessed dataset.')
parser.add_argument('dataset_path', help='Preprocessed file that contains the testing dataset.')
parser.add_argument('-m', '--model_path', default='./model.data', help='Trained model file path.')
parser.add_argument('-v', '--verbose', action='store_true', help='Whether or not print a detailed output.')
parser.add_argument('-i', '--image', action='store_true', help='Whether or not display the images.')
args = parser.parse_args()

def log(message):
    if(args.verbose):
        print(message)

log('loading dataset')
with open(args.dataset_path, 'rb') as f:
    dataset = dill.load(f)

log('loading model')
with open(args.model_path, 'rb') as f:
    model = dill.load(f)

sample_points = model['sample_points']
regressors = model['regressors']
pca_model = model['pca_model']


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

    return item

log('Calculating first estimation')
dataset = list(map(first_estimation, dataset))

def interocular_distance(shape):
    left_eye = shape[114:134]
    right_eye = shape[134:154]

    middle_left = np.mean(left_eye, axis=0)
    middle_right = np.mean(right_eye, axis=0)
    sum_diff = np.sum((middle_left - middle_right) ** 2)
    return np.sqrt(sum_diff)

errors = np.zeros(len(dataset[0]['annotation']))
for j, item in enumerate(dataset):
    image = item['image']
    annotation = item['annotation']

    norm_distance = interocular_distance(annotation)

    for regressor in regressors:
        item['previous_estimation'] = item['estimation']

        for tree in regressor:
            estimation_norm = ((item['estimation']
                            - item['pivot'])
                            / item['scale'])
            params_estimation = pca_model.retrieve_parameters(estimation_norm)

            index = tree.apply(item['intensity_data'])
            prediction = tree.predictions[index] * 0.1
            params_estimation += prediction
        
            new_estimation_norm = pca_model.deform(params_estimation)
            new_estimation = (new_estimation_norm
                        * item['scale']
                        + item['pivot'])
            item['estimation'] = new_estimation
        
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
            except IndexError:
                item['intensity_data'][i] = 0

        if args.image:
            _image = np.copy(image)
            util.plot(_image, item['annotation'], util.BLACK)
            util.plot(_image, item['estimation'], util.WHITE)

            cv2.imshow('image', _image)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                sys.exit(0)
    
    log('Calculating error on {} image {}'.format(j, item['file_name']))
    for i, point_estimation in enumerate(item['estimation']):
        point_annotation = item['annotation'][i]
        distance = np.sqrt(np.sum((point_annotation - point_estimation) ** 2))
        errors[i] += distance / norm_distance

errors /= len(dataset)

print(errors)
print('Average error: {}'.format(np.mean(errors)))