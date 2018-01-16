import numpy as np

class RegressionTree:

    @staticmethod
    def __split_node(node, split_params, split_data):
        left = []
        right = []
        for label in node:
            intensity_u = split_data[label][split_params[0]]
            intensity_v = split_data[label][split_params[1]]
            if abs(intensity_u - intensity_v) > split_params[2]:
                left.append(label)
            else:
                right.append(label)
        return (left, right)


    def __predict_node(self, node, training_data):
        prediction = np.zeros(training_data[node[0]].shape)
        for label in node:
            prediction += training_data[label]
        return prediction / len(node)


    def __calc_split(self, node, training_data, split_data):
        points = np.arange(len(split_data))
        np.random.shuffle(points)
        if len(points) % 2 != 0:
            points = np.delete(points, 0)
        pairs = np.split(points, len(points) / 2)[0:20]
        threshold = np.mean([x - y for x, y in pairs])
        best_pair = pairs[0]
        smallest_error = float("inf")
        for pair in pairs:
            split_params = (pair[0], pair[1], threshold)
            left, right = self.__split_node(node, split_params, split_data)
            prediction_left = self.__predict_node(left, training_data)
            prediction_right = self.__predict_node(right, training_data)
            error = 0
            for label in left:
                diff = np.power(training_data[label] - prediction_left, 2)
                error += np.sum(diff)
            for label in right:
                diff = np.power(training_data[label] - prediction_right, 2)
                error += np.sum(diff)
            if error < smallest_error:
                smallest_error = error
                best_pair = pair
        return (best_pair[0], best_pair[1], threshold)


    def __grow(self, labels, training_data, split_data):
        nodes_queue = [labels]
        levels_queue = [0]
        for _ in range(pow(2, len(self.splits)) - 1):
            node = nodes_queue.pop(0)
            level = levels_queue.pop(0)
            split_params = self.__calc_split(node, training_data, split_data)
            self.splits.append(split_params)
            left, right = self.__split_node(node, split_params, split_data)
            nodes_queue.append(left)
            levels_queue.append(level + 1)

            nodes_queue.append(right)
            levels_queue.append(level + 1)
        for leaf in nodes_queue:
            self.predictions.append(self.__predict_node(leaf, training_data))
            


    def __init__(self, depth, labels, training_data, split_data):
        self.depth = depth
        self.splits = []
        self.predictions = []

        self.__grow(labels, training_data, split_data)
