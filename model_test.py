#!/bin/python3.6
import argparse
import pickle
import dill
import os
import sys
import numpy as np
import cv2
import dlib
import modules.util as util
from imutils import resize
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square

parser = argparse.ArgumentParser(
    description='This script will train a set of regression trees over a preprocessed dataset.')
parser.add_argument(
    'dataset_path', help='Preprocessed file that contains the testing dataset.')
parser.add_argument('-m', '--model_path',
                    default='./model.data', help='Trained model file path.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Whether or not print a detailed output.')
parser.add_argument('-i', '--image', action='store_true',
                    help='Whether or not display the images.')
parser.add_argument('-l', '--limit', default=10,
                    help='Limit the number of regressors to apply.', type=int)
parser.add_argument('--ismuct', default=False, action='store_true')

args = parser.parse_args()


def log(message):
    if(args.verbose):
        print(message)


log('loading dataset')
with open(args.dataset_path, 'rb') as f:
    #dataset = dill.load(f)
    a = args.model_path
    [start, end] = a.split("/")[-1].split("_")[2].split(".")[0].split("-")
    dataset = dill.load(f)[int(start):int(end)]

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
    if args.ismuct:
        left_eye = [shape[27], shape[29]]
        right_eye = [shape[34], shape[32]]
    else:
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

    for regressor in regressors[0:args.limit]:
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
        _image = cv2.imread(
            './datasets/caltech/images/{}'.format(item['file_name']))
        # util.plot(_image, item['annotation'], util.BLACK)
        util.plot(_image, item['estimation'], [0, 255, 255])

        percentage = args.model_path.split('_')[-1]

        # if j in [3, 10, 12, 19]:
        #     name = item['file_name'].replace('.jpg', '_ref.jpg'.format(percentage))
        #     cv2.imwrite(name, _image)

        cv2.imshow('image', resize(_image, height=600))
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            sys.exit(0)

    log('Calculating error on {} image {}'.format(j, item['file_name']))
    for i, point_estimation in enumerate(item['estimation']):
        point_annotation = item['annotation'][i]
        distance = np.sqrt(np.sum((point_annotation - point_estimation) ** 2))
        errors[i] += distance / norm_distance
    log(errors[i])

errors /= len(dataset)

with open("results.txt", "a") as result_file:
    result_file.write(
        '\n{}: {} - {} reg\n'.format(args.model_path, np.mean(errors), args.limit))
