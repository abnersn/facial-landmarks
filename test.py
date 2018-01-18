import cv2
import numpy as np
import dlib
import pickle
import math, os
from imutils import resize
from scipy.spatial.distance import cdist as distance
from modules.data_manager import read_dataset
from modules.regression_tree import RegressionTree
from modules.procrustes import find_theta, rotate

IMAGE_PATH = './img'
DATA_PATH = './data'
SHAPES_MEAN_PATH = 'shapes_mean.data'
REF_POINTS_PATH = 'ref_points.data'
REGRESSOR_PATH = 'regressor.data'
NUMBER_OF_TREES = 500
TREES_DEPTH = 3
NUMBER_OF_REGRESSORS = 1
SHRINKAGE_FACTOR = 0.01

def plot(image, shape):
    radius = int(image.shape[0] * 0.005)
    for i, point in enumerate(shape):
        draw_point = tuple(np.array(point).astype(int))
        cv2.circle(image, draw_point, radius, [255, 255, 255], thickness=-1)

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

with open(SHAPES_MEAN_PATH, 'rb') as f:
    shapes_mean = pickle.load(f)

with open(REF_POINTS_PATH, 'rb') as f:
    ref_points = pickle.load(f)

with open(REGRESSOR_PATH, 'rb') as f:
    regressor = pickle.load(f)

#305917477_2.jpg

training_data = {}

for i, file_name in enumerate(os.listdir(IMAGE_PATH)):
    print(i)
    img = cv2.imread(os.path.join(IMAGE_PATH, file_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    real_shape = np.array(read_dataset(DATA_PATH)[file_name[:-4]])
    faces = detector(img)
    for face in faces:
        top_left = (face.left(), face.top())
        bottom_right = (face.right(), face.bottom())
        middle = ((np.array(top_left) + np.array(bottom_right)) / 2) + np.array([-10, 40])
        scale = face.width() * 0.3
        estimation = (shapes_mean * scale) + middle
        # error = 0
        # for i, point in enumerate(estimation):
        #     real_point = real_shape[i]
        #     diff = np.subtract(point, real_point)
        #     diff = np.power(diff, 2)
        #     diff = np.sum(diff)
        #     error += diff
        # print(error)
        ref_estimation = (ref_points * scale) + middle
        intensity_data = []
        for point in ref_estimation:
            x, y = point.astype(int)
            x = min(x, img.shape[1] - 1)
            y = min(x, img.shape[0] - 1)
            intensity_data.append(img.item(y, x))
        for tree in regressor:
            index = tree.apply(intensity_data)
            estimation = estimation + tree.predictions[index] * 0.01
        # error = 0
        # for i, point in enumerate(estimation):
        #     real_point = real_shape[i]
        #     diff = np.subtract(point, real_point)
        #     diff = np.power(diff, 2)
        #     diff = np.sum(diff)
        #     error += diff
        # print(error)
        training_data[file_name[:-4]] = real_shape - estimation
    #     plot(img, estimation)
    #     cv2.rectangle(img, top_left, bottom_right, 255)

    # img = resize(img, width=400)
    # cv2.imshow('frame', img)
    # key = cv2.waitKey(1000) & 0xFF
    # if key == 27:
    #     break

with open('training_data2.data', 'wb') as f:
    pickle.dump(training_data, f)