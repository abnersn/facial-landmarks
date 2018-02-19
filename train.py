#!/bin/python3.6
import pickle, copy
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from time import sleep
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from multiprocessing import Pool, Process, cpu_count
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

TRAINING_IMAGES = './img_train'
TESTING_IMAGES = './img_test'
ANNOTATIONS_PATH = './data'
NUMBER_OF_TREES = 50
NUMBER_OF_REFPOINTS = 400
TREES_DEPTH = 4
NUMBER_OF_REGRESSORS = 10
SHRINKAGE_FACTOR = 0.05
NUMBER_OF_PARAMETERS = 40
VERBOSE = True
LOAD = True
DISPLAY = True
FONT = cv2.FONT_HERSHEY_SIMPLEX

detector = dlib.get_frontal_face_detector()

def log(message):
    if VERBOSE:
        print(message)

def save(file_name, data):
    log('Saving memory data to disk in file {}'.format(file_name))
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def load(file_name):
    log('Loading file {}'.format(file_name))
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Every image is loaded to the memory
log('Reading images from disk...')
if LOAD:
    images = load('images.bin')
else:
    images = util.read_images(TRAINING_IMAGES)
    save('images.bin', images)

log('Reading annotations from disk...')
if LOAD:
    annotations = load('annotations.bin')
else:
    annotations = util.read_annotations(ANNOTATIONS_PATH)
    save('annotations.bin', annotations)

log('All data has been successfully loaded into memory.')

log('Training face shape model...')
if LOAD:
    model = load('model.bin')
else:
    normalized = calculate_procrustes(annotations)
    model = ShapeModel(NUMBER_OF_PARAMETERS, normalized)
    save('model.bin', model)

log('Sorting sample points...')
if LOAD:
    points = load('points.bin')
else:
    radius = np.max(distance(model.base_shape, model.base_shape)) / 1.7
    points = util.sort_points(NUMBER_OF_REFPOINTS, 0, radius)
    save('points.bin', points)

def first_estimation(item):
    file_name, image = item
    faces = detector(image)
    if len(faces) == 1:

        # Calculate the translation and scale factors of the first estimation
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        translation_factor = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale_factor = faces[0].width() * 0.3

        # Scale and translate the estimation and sample points according
        # to the calculated factors
        estimation = (model.base_shape * scale_factor) + translation_factor
        sample_points = (points * scale_factor) + translation_factor

        # Collect pixel intensity data
        intensity_data = []
        for point in sample_points:
            x = min(int(point[0]), image.shape[1] - 1)
            y = min(int(point[1]), image.shape[0] - 1)
            intensity_data.append(image.item(y, x))
        
        # Retrieve real shape and normalize it to the estimation
        real_shape = annotations[file_name[:-4]]
        real_shape_norm = real_shape - translation_factor
        real_shape_norm /= scale_factor

        # The normalized initial estimation is just the base shape
        estimation_norm = model.base_shape

        # Calculate the parameters that transform the base shape into
        # the normalized versions of both the estimation and the real shape
        params_real_shape = model.retrieve_parameters(real_shape_norm)
        params_estimation = model.retrieve_parameters(estimation_norm)

        # Calculate the regression data
        regression_data = params_real_shape - params_estimation

        return (file_name, {
            'estimation': estimation,
            'sample_points': sample_points,
            'intensity_data': intensity_data,
            'regression_data': regression_data
        })
    else:
        log('No faces or too many faces detected on {}.'.format(file_name))
        os.unlink(os.path.join(TRAINING_IMAGES, file_name))
        return (file_name, None)


p = Pool(cpu_count())
log('Calculating initial data...')
if LOAD:
    data = load('data.bin')
else:
    data = dict(p.map(first_estimation, images.items()))
    save('data.bin', data)


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
        x = min(int(point[0]), image.shape[1] - 1)
        y = min(int(point[1]), image.shape[0] - 1)
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

        sleep(2)
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
