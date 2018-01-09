"""This module manages reading and writing operations of the images dataset.
It reads the Helen dataset as a dictionary with image file names as keys and
a regular python array of points as values. Data is written in a format that
can be read by the corresponding functions.

The facial landmarks in Helen dataset are divided in the following subgroups:

| Index      | Facial Feature   |
| ---------- | ---------------- |
| 0 to 40    | Jawline          |
| 41 to 57   | Nose             |
| 58 to 85   | Mouth (External) |
| 86 to 113  | Mouth (Internal) |
| 114 to 133 | Left Eye         |
| 134 to 153 | Right Eye        |
| 154 to 173 | Left Eyebrow     |
| 174 to 194 | Right Eyebrow    |
"""
__author__ = "Abner S. Nascimento"
__copyright__ = "Copyright 2017, Facial Recognition Project"
__license__ = "Apache"
__version__ = "1.0"
__email__ = "abnersousanascimento@gmail.com"
__status__ = "Development"
import os


def read_dataset(annotations_path):
    """Reads annotations files from dataset.

    Args:
        annotations_path: Path to where the annotations are stored. Defaults
        to current path.

    Returns:
        Dictionary with image file names as keys and an array of points as
        value for each image.
    """
    if not annotations_path:
        annotations_path = "."
    annotation_files = os.listdir(annotations_path)
    images_dictionary = {}

    for annotation_file in annotation_files:
        annotation_file_path = os.path.join(annotations_path, annotation_file)
        with open(annotation_file_path, "r") as csv_file:
            image_file_name = csv_file.readline().rstrip()
            images_dictionary[image_file_name] = []
            for line in csv_file:
                [point_x, point_y] = line.split(" , ")
                point = (float(point_x), float(point_y))
                images_dictionary[image_file_name].append(point)
    return images_dictionary


def write_dataset(dataset, save_directory):
    """Writes a dataset as a format that can be read by read_dataset.

    Args:
        dataset: Dictionary containing the image filenames as keys and shapes
                 as values.
        save_directory: Directory in which to save the dataset files.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    index = 1
    for image_file, points_list in dataset.items():
        file_path = os.path.join(save_directory, "{}.txt".format(index))
        index += 1
        with open(file_path, "w") as file:
            file.write("{}\n".format(image_file))
            for point in points_list:
                file.write("{} , {}\n".format(point[0], point[1]))
