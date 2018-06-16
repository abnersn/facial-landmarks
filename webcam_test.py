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

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img)
    for face in faces:
        top_left = (face.left(), face.top())
        bottom_right = (face.right(), face.bottom())
        pivot = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = face.width() * 0.3

        first_estimation = model.base_shape * scale + pivot
        estimation = first_estimation
        sample_points = sample_points * scale + pivot
        
        intensity_data = []
        for point in sample_points:
            y, x = np.array(point).astype(int)
            try:
                intensity = img.item(x, y)
                intensity_data.append(intensity)
            except IndexError:
                intensity_data.append(0)

        for regressor in regressors:            
            previous_estimation = estimation

            for tree in regressor:
                estimation_norm = (estimation - pivot) / scale
                params_estimation = model.retrieve_parameters(estimation_norm)

                index = tree.apply(intensity_data)
                prediction = tree.predictions[index] * 0.1
                params_estimation += prediction

                new_estimation_norm = model.deform(params_estimation)
                new_estimation = (new_estimation_norm * scale + pivot)
                estimation = new_estimation

            # Update sample points and estimation
            sample_points = util.warp(
                sample_points,
                previous_estimation,
                estimation
            )

            for i, point in enumerate(sample_points):
                y, x = np.array(point).astype(int)
                try:
                    intensity_data[i] = img.item(x, y)
                except IndexError:
                    intensity_data[i] = 0

        util.plot(img, estimation)

    img = resize(img, height=800)
    cv2.imshow('frame', img)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
