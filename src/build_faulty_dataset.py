#!/usr/bin/python
import argparse
import pickle
import dill
import pickle
import numpy as np
import cv2
from modules.util import plot
from modules.face_model import ShapeModel
from modules.procrustes import root_mean_square, calculate_procrustes, find_theta, rotate
from imutils import resize

parser = argparse.ArgumentParser(
    description='Build a faulty dataset and recover the missing points using linear interpolation.')
parser.add_argument(
    'dataset_path', help='Directory to load the pre processed data from.')
parser.add_argument('-c', '--percentage', default=80,
                    help='Percentage of points to eliminate.', type=int)
args = parser.parse_args()

with open(args.dataset_path, 'rb') as f:
    dataset = pickle.load(f)

extremity_points = [0, 14, 15, 45, 37, 21, 38,
                    44, 54, 48, 32, 34, 29, 27, 31, 36, 67, 64]
#extremity_points = [0, 40, 41, 57, 58, 113, 114, 133, 134, 153, 154, 173, 174, 193]

removal_candidates = [i for i in range(0, 76) if i not in extremity_points]
#removal_candidates = [i for i in range(0, 194) if i not in extremity_points]


interpolation_corrected_dataset = []


def correct_interpolate(faulty_shape):
    faulty_groups = [[]]
    for i, point in enumerate(faulty_shape):
        if np.isnan(point).all():
            last_group = faulty_groups[-1]
            if len(last_group) == 0 or last_group[-1] + 1 == i:
                faulty_groups[-1].append(i)
            else:
                faulty_groups.append([i])

    for group in faulty_groups:
        first = min(group)
        if first == 0:
            first = len(faulty_shape) - 1
        else:
            first -= 1
        last = max(group)
        if last == len(faulty_shape) - 1:
            last = 0
        else:
            last += 1
        offset = ((faulty_shape[last] - faulty_shape[first])
                  / (len(group) + 1))
        for i, point_index in enumerate(group):
            faulty_shape[point_index] = (faulty_shape[first]
                                         + offset * (i + 1))
    return faulty_shape


# Remove points from dataset
for i, sample in enumerate(dataset):
    print('Correcting sample {}'.format(i))

    interpolation_corrected = {
        'file_name': sample['file_name'],
        'image': np.copy(sample['image']),
        'annotation': np.copy(sample['annotation']),
        'top_left': np.copy(sample['top_left']),
        'width': sample['width'],
        'height': sample['height']
    }

    total_points = len(sample['annotation'])
    amount_to_remove = int(np.round(total_points * args.percentage / 100))
    faulty_points = np.random.choice(removal_candidates, amount_to_remove)

    faulty_real_shape = np.copy(sample['annotation'])
    faulty_real_shape[faulty_points] = np.array([np.nan, np.nan])

    interpolation_corrected['annotation'] = correct_interpolate(
        faulty_real_shape)

    interpolation_corrected_dataset.append(interpolation_corrected)

print('Saving interpolation corrected dataset')
filename = args.dataset_path
with open(filename + '_{}c'.format(args.percentage), 'wb') as f:
    dill.dump(interpolation_corrected_dataset, f)
