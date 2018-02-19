#!/bin/python3.6
import pickle, copy
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from multiprocessing import Pool, Process, cpu_count
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

TRAINING_IMAGES = './img_train'
TESTING_IMAGES = './img_test'
ANNOTATIONS_PATH = './data'
NUMBER_OF_TREES = 3
NUMBER_OF_REFPOINTS = 400
TREES_DEPTH = 4
NUMBER_OF_REGRESSORS = 2
SHRINKAGE_FACTOR = 0.1
NUMBER_OF_PARAMETERS = 40
VERBOSE = True
LOAD = True
DISPLAY = True

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
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        middle = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = faces[0].width() * 0.3
        return (file_name, {
            'estimation': (model.base_shape * scale) + middle,
            'sample_points': (points * scale) + middle
        })
    else:
        log('No faces or too many faces detected on {}.'.format(file_name))
        os.unlink(os.path.join(TRAINING_IMAGES, file_name))
        return (file_name, None)

def update_data(item):
    file_name, image = item

p = Pool(cpu_count())
log('Calculating initial estimations...')
if LOAD:
    data = load('data.bin')
else:
    data = dict(p.map(first_estimation, images.items()))
    save('data.bin', data)

regressors = []
for r in range(NUMBER_OF_REGRESSORS):
    log('Preprocessing regressor {} data...'.format(r + 1))
    intensity_data = {}
    regression_data = {}
    labels = data.keys()
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

    ###############
    ###############
    ###############
    if r == 0:
        data_before = copy.deepcopy(data)
    ###############
    ###############
    ###############

    trees = []
    for i in range(NUMBER_OF_TREES):
        log('Training tree {}, from regressor {}...'.format(i + 1, r + 1))
        tree = RegressionTree(TREES_DEPTH, labels,
                              regression_data, intensity_data)
        trees.append(tree)

        log('Updating estimations and sample points...')
        count = 0
        for file_name, image in images.items():
            # Normalize estimation
            estimation = data[file_name]['estimation']
            sample_points = data[file_name]['sample_points']
            translation_factor = np.mean(estimation, axis=0)
            estimation_norm = estimation - translation_factor
            scale_factor = root_mean_square(estimation_norm)
            estimation_norm /= scale_factor
            params_estimation = model.retrieve_parameters(estimation_norm)

            test_data = []
            for point in sample_points:
                x = min(int(point[0]), image.shape[1] - 1)
                y = min(int(point[1]), image.shape[0] - 1)
                test_data.append(image.item(y, x))

            index = tree.apply(test_data)
            # print(index, len(tree.predictions))
            delta_params = tree.predictions[index] * SHRINKAGE_FACTOR
            params_estimation += delta_params

            # Update estimations
            new_estimation = model.deform(params_estimation)
            data[file_name]['estimation'] = (new_estimation
                                            * scale_factor
                                            + translation_factor)

            # Update sample points
            data[file_name]['sample_points'] = util.warp(sample_points,
                                                        estimation,
                                                        data[file_name]['estimation'])

            estimation = data[file_name]['estimation']
            real_shape = annotations[file_name[:-4]]

            # Normalize real shape to the current estimation
            real_shape_norm = real_shape - translation_factor
            real_shape_norm /= scale_factor

            # Calculate the parameters that transform the base shape into
            # the normalized versions of both the estimation and the real shape
            params_real_shape = model.retrieve_parameters(real_shape_norm)
            params_estimation = model.retrieve_parameters(estimation_norm)

            # Update intensity data
            intensity_data[file_name] = []
            for point in information['sample_points']:
                x = min(int(point[0]), image.shape[1] - 1)
                y = min(int(point[1]), image.shape[0] - 1)
                intensity_data[file_name].append(image.item(y, x))

            # Update regression data
            regression_data[file_name] = params_real_shape - params_estimation

        count += 1
    regressors.append(trees)
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
