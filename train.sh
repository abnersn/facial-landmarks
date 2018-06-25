#!/bin/bash

echo '20p_40c';
python model_train.py faulty_datasets_interpolation/training_150_20p_40c -o trained_complete_interpolation/model_150_20p_40c -p 120;
echo '20p_60c';
python model_train.py faulty_datasets_interpolation/training_150_20p_60c -o trained_complete_interpolation/model_150_20p_60c -p 120;
echo '20p_80c';
python model_train.py faulty_datasets_interpolation/training_150_20p_80c -o trained_complete_interpolation/model_150_20p_80c -p 120;
