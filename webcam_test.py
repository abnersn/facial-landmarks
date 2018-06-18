#!/bin/python3.6
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

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

with open('reg.data', 'rb') as f:
    regressors = pickle.load(f)

with open('model.data', 'rb') as f:
    model = pickle.load(f)

with open('sample_points.data', 'rb') as f:
    sample_points = pickle.load(f)

item = {}

while(True):
    ret, img = cap.read()
    item['image'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(item['image'])
    for face in faces:
        top_left = np.array([face.left(), face.top()])
        bottom_right = np.array([face.right(), face.bottom()])
        item['pivot'] = (top_left + bottom_right) / 2
        item['scale'] = face.width() * 0.4

        item['estimation'] = model.base_shape * item['scale'] + item['pivot']
        item['sample_points'] = sample_points * item['scale'] + item['pivot']
        
        item['intensity_data'] = []
        for point in item['sample_points']:
            y, x = np.array(point).astype(int)
            try:
                intensity = item['image'].item(x, y)
                item['intensity_data'].append(intensity)
            except IndexError:
                item['intensity_data'].append(0)

        for regressor in regressors:
            item['previous_estimation'] = item['estimation']

            for tree in regressor:
                estimation_norm = ((item['estimation']
                                - item['pivot'])
                                / item['scale'])
                params_estimation = model.retrieve_parameters(estimation_norm)

                index = tree.apply(item['intensity_data'])
                prediction = tree.predictions[index] * 0.1
                params_estimation += prediction
            
                new_estimation_norm = model.deform(params_estimation)
                new_estimation = (new_estimation_norm
                            * item['scale']
                            + item['pivot'])
                item['estimation'] = new_estimation

            # Update sample points and estimation
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
            
        _image = np.copy(item['image'])
        util.plot(_image, item['estimation'], util.WHITE)

        cv2.imshow('image', _image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            sys.exit(0)