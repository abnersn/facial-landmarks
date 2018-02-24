import pickle
import copy
import os
import sys
import numpy as np
import cv2
import dlib
import modules.util as util
from modules.regression_tree import RegressionTree
from modules.face_model import ShapeModel
from multiprocessing import Pool, Process, cpu_count
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square
from scipy.spatial.distance import cdist as distance
from imutils import resize

IMAGE_PATH = './img'
DATA_PATH = './data'
SHAPES_MODEL = 'model.bin'
REF_POINTS_PATH = 'points.bin'
REGRESSOR_PATH = 'regressors_500_10.bin'
SHRINKAGE_FACTOR = 0.02

def plot(image, shape):
    radius = int(image.shape[0] * 0.005)
    for i, point in enumerate(shape):
        draw_point = tuple(np.array(point).astype(int))
        cv2.circle(image, draw_point, radius, [255, 255, 255], thickness=-1)

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

with open(SHAPES_MODEL, 'rb') as f:
    model = pickle.load(f)

with open(REF_POINTS_PATH, 'rb') as f:
    sample_points = pickle.load(f)

with open(REGRESSOR_PATH, 'rb') as f:
    regressors = pickle.load(f)

#305917477_2.jpg

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img)
    for face in faces:
        top_left = (face.left(), face.top())
        bottom_right = (face.right(), face.bottom())
        middle = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = face.width() * 0.3

        test_estimation = (model.base_shape * scale) + middle
        test_sample_points = (sample_points * scale) + middle

        # plot(img, test_sample_points)
        
        for trees in regressors:
            translation_factor = np.mean(test_estimation, axis=0)
            estimation_norm = test_estimation - translation_factor
            scale_factor = root_mean_square(estimation_norm)
            estimation_norm /= scale_factor

            params_estimation = model.retrieve_parameters(estimation_norm)

            test_data = []
            for point in test_sample_points:
                x = min(int(point[0]), img.shape[1] - 1)
                y = min(int(point[1]), img.shape[0] - 1)
                test_data.append(img.item(y, x))

            for tree in trees:
                index = tree.apply(test_data)
                delta_params = tree.predictions[index] * SHRINKAGE_FACTOR
                params_estimation += delta_params
            new_estimation = model.deform(params_estimation)
            new_estimation = (new_estimation * scale_factor
                              + translation_factor)

            # Update sample points
            test_sample_points = util.warp(test_sample_points,
                                           test_estimation,
                                           new_estimation)
            test_estimation = new_estimation

        # plot(img, test_sample_points)
        plot(img, test_estimation)

    img = resize(img, height=800)
    cv2.imshow('frame', img)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
