from modules.regression_tree import RegressionTree
import numpy as np

labels = ['a', 'b', 'c']
training_data = {
    'a': np.array([1, 2, 3]),
    'b': np.array([1, 2, 3]),
    'c': np.array([1, 2, 3])
}
split_data = {
    'a': np.array([1, 2, 3]),
    'b': np.array([10, 20, 30]),
    'c': np.array([100, 200, 300])
}
tree = RegressionTree(3, labels, training_data, split_data)
