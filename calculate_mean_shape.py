from modules.data_manager import read_dataset
from modules.procrustes import calculate_procrustes, mean_of_shapes
from os.path import isdir
import argparse
import pickle
import sys

parser = argparse.ArgumentParser(description='Calculates the mean shape.')
parser.add_argument('directory', help='Directory of landmarks annotations files.')
parser.add_argument('-o', '--output', help='Output file.', default='mean_shape.data')
args = parser.parse_args()

if isdir(args.directory):
    dataset = read_dataset(args.directory)
    normalized = calculate_procrustes(dataset)
    mean = mean_of_shapes(normalized)
    with open(args.output, 'wb') as f:
         pickle.dump(mean, f)
else:
    print('Invalid annotations path.')
    sys.exit(1)
