#!/bin/bash

python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 20 -c 40
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 20 -c 60
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 20 -c 80
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 40 -c 40
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 40 -c 60
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 40 -c 80
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 60 -c 40
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 60 -c 60
python build_faulty_dataset.py datasets/training_300 datasets/training_2000 -p 60 -c 80