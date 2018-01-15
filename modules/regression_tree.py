import numpy as np

class RegressionTree:

    def __split_node(self, node, criteria, intensity_data):
        left = []
        right = []
        for file_name in node:
            intensity_u = intensity_data[file_name][criteria[0]]
            intensity_v = intensity_data[file_name][criteria[1]]
            if intensity_u - intensity_v > criteria[2]:
                left.append(file_name)
            else:
                right.append(file_name)
        return (left, right)


    def __generate_nodes(self, intensity_data):
        nodes_queue = [self.file_names]
        levels_queue = [0]
        for i in range(pow(2, len(self.splits)) - 1):
            node = nodes_queue.pop(0)
            level = levels_queue.pop(0)
            print('splitting node {} by split criteria {}...'.format(i, level))
            left, right = self.__split_node(node,
                                            self.splits[level],
                                            intensity_data)
            nodes_queue.append(left)
            levels_queue.append(level + 1)

            nodes_queue.append(right)
            levels_queue.append(level + 1)
        return nodes_queue

    def grow(self, ref_dataset, est_dataset, intensity_data, shrinkage_factor):
        for leaf in self.__generate_nodes(intensity_data):
            delta_landmarks = np.zeros(est_dataset[0].shape)
            for file_name in leaf:
                real_shape = np.array(ref_dataset, est_dataset[file_name[:-4]])
                delta_landmarks += (real_shape - est_dataset[file_name])
            delta_landmarks = (delta_landmarks / len(leaf))
            self.delta_landmarks.append(shrinkage_factor * delta_landmarks)


    def generate_splits(self, number_of_points):
        for _ in range(self.depth):
            sort_aux = np.arange(number_of_points)
            np.random.shuffle(sort_aux)
            threshold = np.random.randint(255)
            self.splits.append((sort_aux[0], sort_aux[1], threshold))
    def __init__(self, depth, file_names):
        self.depth = depth
        self.file_names = file_names
        self.splits = []
        self.delta_landmarks = []
