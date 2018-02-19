import numpy as np
from scipy.spatial.distance import cdist as distance

class RegressionTree:

    @staticmethod
    def __split_node(node, split_params, data):
        left = []
        right = []
        for label in node:
            intensity_data = data[label]['intensity_data']
            intensity_u = intensity_data[split_params[0]]
            intensity_v = intensity_data[split_params[1]]
            if abs(intensity_u - intensity_v) > split_params[2]:
                left.append(label)
            else:
                right.append(label)
        return (left, right)


    def apply(self, data):
        param_index = 0
        split_params = self.splits[param_index]
        while param_index < (len(self.splits) + 1) / 2:
            index_u, index_v, threshold = split_params
            intensity_u = data[index_u]
            intensity_v = data[index_v]
            if abs(intensity_u - intensity_v) <= threshold:
                param_index += param_index + 1
            else:
                param_index += param_index + 2
            if param_index < (len(self.splits) + 1) / 2:
                split_params = self.splits[param_index]
        return int(param_index - (len(self.splits) + 1) / 2)

    def __predict_node(self, node, data):
        if len(node) == 0:
            return np.zeros(self.shape)

        prediction = np.zeros(self.shape)
        for label in node:
            regression_data = data[label]['regression_data']
            prediction += regression_data
        return prediction / len(node)


    def __calc_split(self, node, data):

        point_u, point_v = self.pairsQueue.pop(0)

        thresholds = np.random.randint(0, 255, 20)

        maximum_diff = -float("inf")
        best_threshold = 0

        for threshold in thresholds:
            split_params = (point_u, point_v, threshold)
            left, right = self.__split_node(node, split_params, data)
            prediction_left = self.__predict_node(left, data)
            prediction_right = self.__predict_node(right, data)
            diff = np.sum(
                np.dot(prediction_left, prediction_left.T) * len(left)
                + np.dot(prediction_right, prediction_right.T) * len(right)
            )
            if diff > maximum_diff:
                maximum_diff = diff
                best_threshold = threshold

        return (point_u, point_v, best_threshold)


    def __grow(self, labels, data):
        nodes_queue = [labels]
        levels_queue = [0]
        for _ in range(pow(2, (self.depth - 1)) - 1):
            node = nodes_queue.pop(0)
            level = levels_queue.pop(0)
            split_params = self.__calc_split(node, data)
            self.splits.append(split_params)
            left, right = self.__split_node(node, split_params, data)
            nodes_queue.append(left)
            levels_queue.append(level + 1)

            nodes_queue.append(right)
            levels_queue.append(level + 1)
        for leaf in nodes_queue:
            self.predictions.append(self.__predict_node(leaf, data))
            


    def __init__(self, depth, labels, data):
        self.depth = depth
        self.splits = []
        self.predictions = []

        # A key to help getting a sample from the data
        key = next(iter(data))
        sample = data[key]

        self.shape = sample['regression_data'].shape
        self.number_of_points = len(sample['intensity_data'])

        # Retrieve both the minimum and maximum intensity difference value
        self.pairsQueue = []
        distances = distance(sample['sample_points'], sample['sample_points'])
        # Sort 20 random points
        points_u_indexes = np.random.randint(0, self.number_of_points, 20)
        for index in points_u_indexes:
            roulette = distances[index] / np.sum(distances[index])
            sorted_random = np.random.rand()
            aux_sum = 0
            chosen_index = 0
            while aux_sum < sorted_random:
                aux_sum += roulette[chosen_index]
                chosen_index += 1
            self.pairsQueue.append([index, chosen_index])

        self.__grow(labels, data)
