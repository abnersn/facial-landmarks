import pickle
import copy
import os
import sys, copy
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

with open('reg.bin', 'rb') as f:
    regressors = pickle.load(f)

with open('model.bin', 'rb') as f:
    model = pickle.load(f)

with open('sample_points.bin', 'rb') as f:
    sample_points = pickle.load(f)

def warp2(shape_a, shape_b, groups):
    diff = shape_a - shape_b
    return np.array([group + diff[i] for i, group in enumerate(groups)])

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img)
    for face in faces:
        top_left = (face.left(), face.top())
        bottom_right = (face.right(), face.bottom())
        pivot = ((np.array(top_left) + np.array(bottom_right)) / 2)
        scale = face.width() * 0.3

        test_estimation = model.base_shape * scale + pivot
        test_sample_points = sample_points * scale + pivot
        
        for regressor in regressors:

            new_img = copy.copy(img)
            util.plot(new_img, test_estimation)
            new_img = resize(new_img, height=800)
            cv2.imshow('frame', new_img)
            cv2.waitKey(0)

            intensity_data = []
            for group in test_sample_points:
                intensity_group = []
                for point in group:
                    y, x = np.array(point).astype(int)
                    try:
                        intensity = img.item(x, y)
                        intensity_group.append(intensity)
                    except IndexError:
                        intensity_group.append(-1)
                intensity_data.append(intensity_group)

            test_estimation_norm = (test_estimation - pivot) / scale
            params_estimation = model.retrieve_parameters(test_estimation_norm)
            print(params_estimation)
            for tree in regressor:
                index = tree.apply(intensity_data)
                delta_params = tree.predictions[index]
                params_estimation += delta_params * 0.01
            print(params_estimation)
            new_estimation = model.deform(params_estimation)
            new_estimation = (new_estimation * scale + pivot)

            # Update sample points and estimation
            test_sample_points = warp2(test_estimation, new_estimation, test_sample_points)
            test_estimation = new_estimation

        
        # util.plot(new_img, test_sample_points.flatten().reshape([3 * len(model.base_shape), 2]))
