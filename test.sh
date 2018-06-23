#!/bin/bash

# ./model_test.py datasets/testing -m model_150i_70p;
# ./model_test.py datasets/testing -m model_150i_120p;

# ./model_test.py datasets/testing -m model_300i_70p;
# ./model_test.py datasets/testing -m model_300i_120p;

# ./model_test.py datasets/testing -m model_450i_70p;
# ./model_test.py datasets/testing -m model_450i_120p;

# ./model_test.py datasets/testing -m model_2000i_70p;
# ./model_test.py datasets/testing -m model_2000i_120p;


# Average error: 0.21342316317133106
# Average error: 0.1758377357522884
# Average error: 0.20808439890984962
# Average error: 0.1633438050196182
# Average error: 0.20206613437269869
# Average error: 0.14677045366443095
# Average error: 0.19361847564868384
# Average error: 0.10895693176874148

python model_test.py datasets/testing -m trained_complete_pca/model_300_20p_40c;
python model_test.py datasets/testing -m trained_complete_pca/model_300_20p_60c;
python model_test.py datasets/testing -m trained_complete_pca/model_300_20p_80c;


python model_test.py datasets/testing -m trained_complete_pca/model_300_40p_40c;
python model_test.py datasets/testing -m trained_complete_pca/model_300_40p_60c;
python model_test.py datasets/testing -m trained_complete_pca/model_300_40p_80c;

python model_test.py datasets/testing -m trained_complete_pca/model_300_60p_40c;
python model_test.py datasets/testing -m trained_complete_pca/model_300_60p_60c;
python model_test.py datasets/testing -m trained_complete_pca/model_300_60p_80c;

# Average error: 0.11956137514809759
# Average error: 0.13144693464676266
# Average error: 0.13809207301655935
# Average error: 0.12169936975789727
# Average error: 0.1298734424243041
# Average error: 0.1343389211304887
# Average error: 0.125051862842833
# Average error: 0.1307528132317951
# Average error: 0.14232084426104086