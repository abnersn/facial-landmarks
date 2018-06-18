#!/bin/bash

./model_test.py datasets/testing -m model_150i_70p;
./model_test.py datasets/testing -m model_150i_120p;

./model_test.py datasets/testing -m model_300i_70p;
./model_test.py datasets/testing -m model_300i_120p;

./model_test.py datasets/testing -m model_450i_70p;
./model_test.py datasets/testing -m model_450i_120p;

./model_test.py datasets/testing -m model_2000i_70p;
./model_test.py datasets/testing -m model_2000i_120p;


# Average error: 0.21342316317133106
# Average error: 0.1758377357522884
# Average error: 0.20808439890984962
# Average error: 0.1633438050196182
# Average error: 0.20206613437269869
# Average error: 0.14677045366443095
# Average error: 0.19361847564868384
# Average error: 0.10895693176874148
