import numpy as np
from scipy.spatial.distance import cdist as distance
from modules.procrustes import calculate_procrustes, mean_of_shapes, root_mean_square

class RegressionTree:

    @staticmethod
    def __split_node(node, split_params):
        threshold, point_u, point_v = split_params        
        left = []
        right = []
        for sample in node:
            intensity_data = sample['intensity_data']
            intensity_u = intensity_data[point_u]
            intensity_v = intensity_data[point_v]
            if intensity_u - intensity_v > threshold:
                left.append(sample)
            else:
                right.append(sample)
        return [left, right]

    def __split_sample(self, intensity_data, split_params):
        threshold, point_u, point_v = split_params
        intensity_u = intensity_data[point_u]
        intensity_v = intensity_data[point_v]
        if intensity_u - intensity_v > threshold:
            return 0
        else:
            return 1


    def apply(self, sample):
        index = 0
        for i, split in enumerate(self.splits):
            factor = self.__split_sample(sample, split)
            index += factor * (2 ** (len(self.splits) - i - 1))
        return index
            

    def __predict_node(self, node):
        if len(node) == 0:
            return np.zeros(self.shape)

        prediction = np.zeros(self.shape)
        for sample in node:
            prediction += sample['regression_data']
        return prediction / len(node)


    def __grow(self, dataset):
        nodes = [dataset]
        for i, split in enumerate(self.splits):
            thresholds = np.random.randint(-255, 255, 20)
            maximum_diff = -float("inf")
            best_threshold = thresholds[0]

            # Calculate the best thresholds
            for threshold in thresholds:
                diff = 0
                split[0] = threshold
                for node in nodes:
                    left, right = self.__split_node(node, split)
                    prediction_left = self.__predict_node(left)
                    prediction_right = self.__predict_node(right)
                    diff += (len(left) * np.sum(prediction_left ** 2)
                           + len(right) * np.sum(prediction_right ** 2))
                if diff > maximum_diff:
                    maximum_diff = diff
                    best_threshold = threshold

            self.splits[i][0] = best_threshold
            
            new_nodes = []
            for node in nodes:
                left, right = self.__split_node(node, self.splits[i])
                new_nodes.append(left)
                new_nodes.append(right)
            nodes = new_nodes
        
        for node in nodes:
            self.predictions.append(self.__predict_node(node))


    def __select_point(self, base_point, points):
        scores = np.sum((points - base_point) ** 2, axis=1).astype(np.float32)
        scores = [score if score != 0 else np.inf for score in scores]
        scores = np.array(scores) ** -1
        scores /= np.sum(scores)
        acc = 0
        r = np.random.ranf()
        for i, score in enumerate(scores):
            if (score > 0):
                acc += score
                if acc > r:
                    return i
        return np.random.randint(0, len(points))


    def __init__(self, depth, dataset):
        self.predictions = []
        self.depth = depth

        sample = dataset[0]
        self.shape = sample['regression_data'].shape
        self.splits = []
        for _ in range(depth):
            index_u = np.random.randint(len(sample['sample_points']))
            point_u = sample['sample_points'][index_u]
            index_v = self.__select_point(point_u, sample['sample_points'])
            self.splits.append([0, index_u, index_v])

        self.__grow(dataset)
