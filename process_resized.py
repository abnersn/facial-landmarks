#!/bin/python3.6
import os, sys
import numpy as np
import cv2

print('reading annotations')
for annotation_file in os.listdir('./datasets/helen/annotations/'):
    path = os.path.join('./datasets/helen/annotations/', annotation_file)
    with open(path, 'r') as csv_file:

        # read the annotations
        image_file = csv_file.readline().rstrip() + '.jpg'
        points = []
        for line in csv_file:
            [point_x, point_y] = line.split(' , ')
            point = (float(point_x), float(point_y))
            points.append(point)

        # load the resized and original images
        original_path = os.path.join('./datasets/helen/original', image_file)
        resized_path = os.path.join('./datasets/helen/resized', image_file)
        original = cv2.imread(original_path, 0)
        resized = cv2.imread(resized_path, 0)
        ratio = np.mean(np.array(original.shape) / np.array(resized.shape))
        annotations = (np.array(points) / ratio).astype(np.uint16)

        with open('./datasets/helen/resized_annotations/{}.txt'.format(image_file[0:-4]), 'w+') as annotation_file:
            for point in annotations:
                annotation_file.write('{} , {}\n'.format(point[0], point[1]))

        # if ratio > 1:
        #     print('was resized')
        # for point in annotations:
        #     cv2.circle(resized, tuple(point), 2, [222,222,222])
        # cv2.imshow('Image', resized)
        # cv2.waitKey(0)



print('finished processing')