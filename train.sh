#!/bin/bash

./model_train.py datasets/training_150 -o model_150i_60p -p 70;
./model_train.py datasets/training_150 -o model_150i_90p -p 120;

./model_train.py datasets/training_150 -o model_300i_70p -p 70;
./model_train.py datasets/training_150 -o model_300i_120p -p 120;

./model_train.py datasets/training_150 -o model_450i_70p -p 70;
./model_train.py datasets/training_150 -o model_450i_120p -p 120;

./model_train.py datasets/training_150 -o model_2000i_70p -p 70;
./model_train.py datasets/training_150 -o model_2000i_120p -p 120;