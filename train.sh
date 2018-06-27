#!/bin/bash

echo '40c';
python model_train.py faulty_datasets_interpolation/training_150_20p_40c -o trained_complete_interpolation/model_150_90p_40c -p 90;
python model_train.py faulty_datasets_interpolation/training_150_20p_40c -o trained_complete_interpolation/model_150_70p_40c -p 70;
echo '60c';
python model_train.py faulty_datasets_interpolation/training_150_20p_60c -o trained_complete_interpolation/model_150_90p_60c -p 90;
python model_train.py faulty_datasets_interpolation/training_150_20p_60c -o trained_complete_interpolation/model_150_70p_60c -p 70;
echo '80c';
python model_train.py faulty_datasets_interpolation/training_150_20p_80c -o trained_complete_interpolation/model_150_90p_80c -p 90;
python model_train.py faulty_datasets_interpolation/training_150_20p_80c -o trained_complete_interpolation/model_150_70p_80c -p 70;
