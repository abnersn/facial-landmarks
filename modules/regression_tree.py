class RegressionTree:
    def split_node(self, split_data, intensity_data):
        left = []
        right = []
        for file_name in self.file_names:
            intensity_u = intensity_data[file_name][split_data[0]]
            intensity_v = intensity_data[file_name][split_data[1]]
            if intensity_u - intensity_v > split_data[2]:
                left.append(file_name)
            else:
                right.append(file_name)
        return (left, right)

    def grow(self, data):
        nodes_queue = [self.file_names]
        levels_queue = [0]
        for i in range(pow(2, len(splits)) - 1):
            node = nodes_queue.pop(0)
            level = levels_queue.pop(0)
            print('splitting node {} by split criteria {}...'.format(i, level))
            left, right = split_node(node, splits[level], intensity_data)
            nodes_queue.append(left)
            levels_queue.append(level + 1)

            nodes_queue.append(right)
            levels_queue.append(level + 1)
        return nodes_queue
        for leaf in grow_tree(files, tree_splits, data):
            delta_landmarks = np.zeros(shapes_mean.shape)
            for file_name in leaf:
                real_shape = np.array(dataset[file_name[:-4]])
                s, _, t = similarity_transform(real_shape, shapes_mean)
                estimation = (shapes_mean / s) + t
                delta_landmarks += real_shape - estimation
            delta_landmarks = SHRINKAGE_FACTOR * (delta_landmarks / len(leaf))
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
