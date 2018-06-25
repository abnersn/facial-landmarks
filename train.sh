#!/bin/bash

echo '20p_40c';
python model_train.py faulty_datasets_interpolation/training_300_20p_40c -o trained_complete_interpolation/model_300_20p_40c -p 120;
echo '20p_60c';
python model_train.py faulty_datasets_interpolation/training_300_20p_60c -o trained_complete_interpolation/model_300_20p_60c -p 120;
echo '20p_80c';
python model_train.py faulty_datasets_interpolation/training_300_20p_80c -o trained_complete_interpolation/model_300_20p_80c -p 120;
echo '40p_40c';
python model_train.py faulty_datasets_interpolation/training_300_40p_40c -o trained_complete_interpolation/model_300_40p_40c -p 120;
echo '40p_60c';
python model_train.py faulty_datasets_interpolation/training_300_40p_60c -o trained_complete_interpolation/model_300_40p_60c -p 120;
echo '40p_80c';
python model_train.py faulty_datasets_interpolation/training_300_40p_80c -o trained_complete_interpolation/model_300_40p_80c -p 120;
echo '60p_40c';
python model_train.py faulty_datasets_interpolation/training_300_60p_40c -o trained_complete_interpolation/model_300_60p_40c -p 120;
echo '60p_60c';
python model_train.py faulty_datasets_interpolation/training_300_60p_60c -o trained_complete_interpolation/model_300_60p_60c -p 120;
echo '60p_80c';
python model_train.py faulty_datasets_interpolation/training_300_60p_80c -o trained_complete_interpolation/model_300_60p_80c -p 120;