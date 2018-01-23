#!/bin/python3.6
import pickle
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from modules.regression_tree import RegressionTree
from multiprocessing import Pool
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

TRAINING_IMAGES = './dev_test'
TESTING_IMAGES = './dev_test'
ANNOTATIONS_PATH = './data'
NUMBER_OF_TREES = 50
NUMBER_OF_REFPOINTS = 40
TREES_DEPTH = 3
NUMBER_OF_REGRESSORS = 1
SHRINKAGE_FACTOR = 0.01

detector = dlib.get_frontal_face_detector()

# Every image is loaded into the memory
print('reading images from disk')
images = util.read_images(TRAINING_IMAGES)
print('reading annotations from disk')
annotations = util.read_annotations(ANNOTATIONS_PATH)
print('all data has been successfully loaded into memory')

print('calculating mean of shapes...')
normalized = calculate_procrustes(annotations)
shapes_mean = mean_of_shapes(normalized)

print('sorting sample points...')
radius = np.max(distance(shapes_mean, shapes_mean)) / 2
points = util.sort_points(NUMBER_OF_REFPOINTS, 0, radius)

def first_estimation(item):
    file_name, image = item
    faces = detector(image)
    if len(faces) == 1:
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        middle = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = faces[0].width() * 0.3
        return (file_name, {
            'estimation': (shapes_mean * scale) + middle,
            'sample_points': (points * scale) + middle
        })
    else:
        print('no face or too many faces detected on {}'.format(file_name))
        os.unlink(os.path.join(TRAINING_IMAGES, file_name))
        return (file_name, None)

p = Pool(4)
print('calculating initial estimations...')
data = dict(p.map(first_estimation, images.items()))

print('showing images...')
for file_name, information in data.items():
    image = images[file_name]

    estimation = data[file_name]['estimation']
    real_shape = annotations[file_name[:-4]]

    # Normalizes both shapes to current estimation's vector space
    translation_factor = np.mean(estimation, axis=0)
    estimation -= translation_factor
    real_shape -= translation_factor
    scale_factor = root_mean_square(estimation)
    estimation /= scale_factor
    real_shape /= scale_factor


    util.plot(image, real_shape)
    util.plot(image, estimation)
    cv2.imshow('image', resize(image, width=400))
    cv2.waitKey(300)
input('halt...')

##################################################

difference_data = {}
intensity_data = {}
labels = []
for file_name, information in data.items():
    labels.append(file_name)
    image = images[file_name]
    real_shape = annotations[file_name[:-4]] 
    difference_data[file_name] = (real_shape - information['estimation'])
    intensity_data[file_name] = []    
    for point in information['sample_points']:
        x = min(int(point[0]), image.shape[1] - 1)
        y = min(int(point[1]), image.shape[0] - 1)
        intensity_data[file_name].append(image.item(y, x))

trees = []
for i in range(NUMBER_OF_TREES):
    print('training tree {}...'.format(i))
    trees.append(RegressionTree(TREES_DEPTH, labels, difference_data, intensity_data))

for file_name in os.listdir(TESTING_IMAGES):
    img = cv2.imread(os.path.join(TESTING_IMAGES, file_name), 0)
    faces = detector(img)
    if len(faces) > 0:
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        cv2.rectangle(img, top_left, bottom_right, util.WHITE)
        middle = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = faces[0].width() * 0.3
        estimation = (shapes_mean * scale) + middle
        sample_points = (points * scale) + middle
        test_data = []
        for point in sample_points:
            x = min(int(point[0]), img.shape[1] - 1)
            y = min(int(point[1]), img.shape[0] - 1)
            test_data.append(img.item(y, x))
        for tree in trees:
            index = tree.apply(test_data)
            delta = tree.predictions[index]
            estimation = estimation + SHRINKAGE_FACTOR * delta
    util.plot(img, estimation)
    cv2.imshow('image', resize(img, width=400))
    cv2.waitKey(300)

input('halt...')