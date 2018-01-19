#!/bin/python3.6
import pickle
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from multiprocessing import Pool
from modules.procrustes import calculate_procrustes, mean_of_shapes
from scipy.spatial.distance import cdist as distance
from imutils import resize

IMAGES_PATH = './img_test'
ANNOTATIONS_PATH = './data'
NUMBER_OF_TREES = 500
NUMBER_OF_REFPOINTS = 400
TREES_DEPTH = 3
NUMBER_OF_REGRESSORS = 1
SHRINKAGE_FACTOR = 0.01

detector = dlib.get_frontal_face_detector()

# Every image is loaded into the memory
print('reading images from disk')
images = util.read_images(IMAGES_PATH)
print('reading annotations from disk')
annotations = util.read_annotations(ANNOTATIONS_PATH)
print('all data has been successfully loaded into memory')

print('calculating mean of shapes...')
annotations = calculate_procrustes(annotations)
shapes_mean = mean_of_shapes(annotations)

print('sorting reference points')
radius = np.max(distance(shapes_mean, shapes_mean)) / 2
points = util.sort_points(400, 0, radius)

def first_estimation(item):
    file_name, image = item
    faces = detector(image)
    if len(faces) > 0:
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        middle = ((np.array(top_left) + np.array(bottom_right))
                  / 2) + np.array([-10, 40])
        scale = faces[0].width() * 0.3
        return (file_name, {
            'estimation': (shapes_mean * scale) + middle,
            'reference_points': (points * scale) + middle
        })
    else:
        print('no face detected on {}'.format(file_name))
        os.unlink(os.path.join(IMAGES_PATH, file_name))
        return (file_name, None)

p = Pool(4)
print('calculating estimations...')
data = dict(p.map(first_estimation, images.items()))

# print('showing images...')
# for file_name, information in data:
#     image = images[file_name]
#     util.plot(image, information['reference_points'])
#     cv2.imshow('image', resize(image, width=400))
#     cv2.waitKey(300)

difference_data = {}
intensity_data = {}
for file_name, information in data.items():
    image = images[file_name]
    real_shape = annotations[file_name[:-4]] 
    difference_data[file_name] = (real_shape - information['estimation'])
    intensity_data[file_name] = []    
    for point in information['reference_points']:
        x = min(int(point[0]), image.shape[1] - 1)
        y = min(int(point[1]), image.shape[0] - 1)
        intensity_data[file_name].append(image.item(y, x))

print(intensity_data)
input('halt...')