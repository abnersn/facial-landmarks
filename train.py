#!/bin/python3.6
import pickle
import os, sys
import numpy as np
import cv2, dlib
import modules.util as util
from multiprocessing import Pool
from modules.procrustes import calculate_procrustes, mean_of_shapes


IMAGES_PATH = './img_train'
ANNOTATIONS_PATH = './data'
NUMBER_OF_TREES = 500
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

annotations = calculate_procrustes(annotations)
shapes_mean = mean_of_shapes(annotations)


def calculate_estimation(item):
    file_name, image = item
    faces = detector(image)
    if len(faces) > 0:
        top_left = (faces[0].left(), faces[0].top())
        bottom_right = (faces[0].right(), faces[0].bottom())
        middle = ((np.array(top_left) + np.array(bottom_right))
                  / 2) + np.array([-10, 40])
        scale = faces[0].width() * 0.3
        estimation = (shapes_mean * scale) + middle
        return (file_name, estimation)
    else:
        print('no face detected on {}'.format(file_name))
        os.unlink(os.path.join(IMAGES_PATH, file_name))
        return (file_name, [])

p = Pool(4)
estimations = p.map(calculate_estimation, images.items())

for file_name, estimation in estimations:
    util.plot(images[file_name], estimation)
    cv2.imshow('image', images[file_name])
    cv2.waitKey(1000)

input('halt...')

print(shapes_mean)