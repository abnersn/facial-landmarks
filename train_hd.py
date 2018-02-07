#!/bin/python3.6
import pickle
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from multiprocessing import Pool
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

TRAINING_IMAGES = './img_train'
TESTING_IMAGES = './img_test'
ANNOTATIONS_PATH = './data'
NUMBER_OF_TREES = 500
NUMBER_OF_REFPOINTS = 400
TREES_DEPTH = 3
NUMBER_OF_REGRESSORS = 1
SHRINKAGE_FACTOR = 0.001
NUMBER_OF_PARAMETERS = 30
VERBOSE = True

detector = dlib.get_frontal_face_detector()

def log(message):
    if VERBOSE:
        print(message)

log('Reading images from disk...')
with open('images.bin', 'rb') as f:
    images = pickle.load(f)

log('Reading annotations from disk...')
with open('annotations.bin', 'rb') as f:
    annotations = pickle.load(f)

log('Reading data from disk...')
with open('data.bin', 'rb') as f:
    data = pickle.load(f)

log('Reading face model from disk...')
with open('model.bin', 'rb') as f:
    model = pickle.load(f)

log('Sorting sample points...')
radius = np.max(distance(model.base_shape, model.base_shape)) / 2
points = util.sort_points(NUMBER_OF_REFPOINTS, 0, radius)

labels = data.keys()

log('Preprocessing regression data...')
intensity_data = {}
regression_data = {}
for file_name, information in data.items():
    image = images[file_name]

    estimation = data[file_name]['estimation']
    real_shape = annotations[file_name[:-4]]

    # Normalize shapes according to the current estimation
    translation_factor = np.mean(estimation, axis=0)
    estimation_norm = estimation - translation_factor
    real_shape_norm = real_shape - translation_factor
    scale_factor = root_mean_square(estimation_norm)
    estimation_norm /= scale_factor
    real_shape_norm /= scale_factor

    # Calculate the parameters that transform the base shape into
    # the normalized versions of both the estimation and the real shape
    params_real_shape = model.retrieve_parameters(real_shape_norm)
    params_estimation = model.retrieve_parameters(estimation_norm)

    # Organize the intensity data into a dictionary
    intensity_data[file_name] = []    
    for point in information['sample_points']:
        x = min(int(point[0]), image.shape[1] - 1)
        y = min(int(point[1]), image.shape[0] - 1)
        intensity_data[file_name].append(image.item(y, x))
    
    # Organize the regression data into a dictionary
    regression_data[file_name] = params_real_shape - params_estimation

    test_real_shape = model.deform(params_real_shape)
    test_real_shape = test_real_shape * scale_factor + translation_factor
    test_estimation = model.deform(params_estimation)
    test_estimation = test_estimation * scale_factor + translation_factor

    # color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # util.plot(color, test_real_shape, [255, 20, 20])
    # util.plot(color, test_estimation, [20, 20, 255])
    # cv2.imshow('image', resize(color, width=400))
    # cv2.waitKey(300)

##################################################

trees = []
for i in range(NUMBER_OF_TREES):
    log('training tree {}...'.format(i))
    trees.append(RegressionTree(TREES_DEPTH, labels, regression_data, intensity_data))

total_error_decrease = 0
average_decrease_percent = 0
total = len(os.listdir(TESTING_IMAGES))
for file_name in os.listdir(TESTING_IMAGES):
    real_shape = annotations[file_name[:-4]]
    img = cv2.imread(os.path.join(TESTING_IMAGES, file_name), 0)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    faces = detector(img)
    if len(faces) > 0:
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        cv2.rectangle(img, top_left, bottom_right, util.WHITE)
        middle = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = faces[0].width() * 0.3
        estimation = (model.base_shape * scale) + middle
        sample_points = (points * scale) + middle
        test_data = []
        for point in sample_points:
            x = min(int(point[0]), img.shape[1] - 1)
            y = min(int(point[1]), img.shape[0] - 1)
            test_data.append(img.item(y, x))

        # Normalize estimation
        translation_factor = np.mean(estimation, axis=0)
        estimation_norm = estimation - translation_factor
        scale_factor = root_mean_square(estimation_norm)
        estimation_norm /= scale_factor
        params_estimation = model.retrieve_parameters(estimation_norm)

        for tree in trees:
            index = tree.apply(test_data)
            delta_params = tree.predictions[index] * SHRINKAGE_FACTOR
            params_estimation += delta_params
        
        test_estimation = model.deform(params_estimation)
        test_estimation = test_estimation * scale_factor + translation_factor

        # Calculate the error for first estimation
        error_first_estimation = 0
        for i, point in enumerate(estimation):
            difference = point - real_shape[i]
            square = np.power(difference, 2)
            error_first_estimation += np.sum(square)
        print('First estimation: ', error_first_estimation)

        # Calculate the error for current estimation
        error_current = 0
        for i, point in enumerate(test_estimation):
            difference = point - real_shape[i]
            square = np.power(difference, 2)
            error_current += np.sum(square)
        print('Current Estimation', error_current)

        if error_current < error_first_estimation:
            print('ok')
            total_error_decrease += 1
            percent = (error_first_estimation - error_current) / error_first_estimation
            average_decrease_percent += percent
        else:
            print('not ok')

        util.plot(color, test_estimation, [255, 0 ,0])
        util.plot(color, estimation, [25, 255,0])
    cv2.imshow('image', resize(color, height=800))
    key = cv2.waitKey() & 0xFF
    if key == 27:
        break
    elif key == 110:
        continue
print('Total: ', total)
print('Error decrease: ', total_error_decrease)
print('Percentage: ', total_error_decrease/total * 100)
print('Average decrease percentage: ', average_decrease_percent/total * 100)
input('halt...')