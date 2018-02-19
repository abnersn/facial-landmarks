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

    def __predict_node(self, node, training_data):
        if len(node) == 0:
            return np.zeros(self.shape)

        prediction = np.zeros(self.shape)
        for label in node:
            prediction += training_data[label]
        return prediction / len(node)


    def __calc_split(self, node, training_data, split_data):
        maximum_diff = 0
        best_threshold = 0

        point_u = self.pointsQueue.pop(0)
        point_v = self.pointsQueue.pop(0)

        thresholds = np.random.randint(self.minIntensity, self.maxIntensity, 20)

        for threshold in thresholds:
            split_params = (point_u, point_v, threshold)
            left, right = self.__split_node(node, split_params, split_data)
            prediction_left = self.__predict_node(left, training_data)
            prediction_right = self.__predict_node(right, training_data)
            diff = np.sum(
                np.dot(prediction_left, prediction_left.T) * len(left)
                + np.dot(prediction_right, prediction_right.T) * len(right)
            )
            if diff > maximum_diff:
                maximum_diff = diff
                best_threshold = threshold

        return (point_u, point_v, best_threshold)


    def __grow(self, labels, training_data, split_data):
        nodes_queue = [labels]
        levels_queue = [0]
        for _ in range(pow(2, (self.depth - 1)) - 1):
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
        self.shape = training_data[list(labels)[0]].shape

        self.minIntensity = np.min(list(split_data.values()))
        self.maxIntensity = np.max(list(split_data.values()))
        
        key = next(iter(split_data))
        numberOfPoints = len(split_data[key])
        self.pointsQueue = list(np.arange(numberOfPoints))
        np.random.shuffle(self.pointsQueue)

        self.__grow(labels, training_data, split_data)
