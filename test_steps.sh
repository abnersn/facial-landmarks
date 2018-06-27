#!/bin/bash

# python model_test.py datasets/testing -m trained_complete/model_2000i_70p
# python model_test.py datasets/testing -m trained_complete/model_2000i_90p
# python model_test.py datasets/testing -m trained_complete/model_2000i_120p

# R0
# Average error: 0.5070395276683588
# Average error: 0.5070395276683588
# Average error: 0.5070395276683588

# R1
# Average error: 0.19993884052198488
# Average error: 0.19922609040403674
# Average error: 0.15159750803031113

# R2
# Average error: 0.19517035311805098
# Average error: 0.19356019330568452
# Average error: 0.12492888844855202

# R3
# Average error: 0.19410406646923692
# Average error: 0.19237319363325023
# Average error: 0.11694164950669836

# R4
# Average error: 0.1937097806465948
# Average error: 0.19159132894463352
# Average error: 0.1132251448790941

# R5
# Average error: 0.19356265456187305
# Average error: 0.1914000939319602
# Average error: 0.11083213590877054

# R6
# Average error: 0.19362711701862284
# Average error: 0.19124625232605763
# Average error: 0.10983590604019007

# R7
# Average error: 0.1935926990052798
# Average error: 0.19130737493529468
# Average error: 0.10932429551529636

# R8
# Average error: 0.1936042405280114
# Average error: 0.19129500703722638
# Average error: 0.10921510413066005

# R9
# Average error: 0.19352971203731748
# Average error: 0.191305983745947
# Average error: 0.10901898032245518

# R10
# Average error: 0.19361847564868384
# Average error: 0.19134632749093416
# Average error: 0.10895693176874148

# Interpolado 40%
# Average error: 0.1541558368909784
# Average error: 0.12826118473821535
# Average error: 0.12028889062236606
# Average error: 0.11685496681140639
# Average error: 0.11484620331871179
# Average error: 0.11402253025915975
# Average error: 0.113636533829164
# Average error: 0.11340926492702358
# Average error: 0.1133095947672056
# Average error: 0.11327588263565937

python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 1
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 2
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 3
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 4
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 5
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 6
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 7
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 8
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 9
python model_test.py datasets/testing -m trained_complete_interpolation/model_150_20p_80c -l 10