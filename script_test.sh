#!/bin/bash

for LIMIT in {0..10}
do
# HELEN
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data -l $LIMIT

cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data_80c -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data_80c -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data_80c -l $LIMIT

cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data_60c -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data_60c -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data_60c -l $LIMIT

cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data_40c -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data_40c -l $LIMIT
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data_40c -l $LIMIT

# MUCT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data --ismuct -l $LIMIT

cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data_80c --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data_80c --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data_80c --ismuct -l $LIMIT

cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data_60c --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data_60c --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data_60c --ismuct -l $LIMIT

cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data_40c --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data_40c --ismuct -l $LIMIT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data_40c --ismuct -l $LIMIT
done