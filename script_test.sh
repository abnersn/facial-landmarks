#!/bin/bash

# HELEN
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data

cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data_80c
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data_80c
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data_80c

cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data_60c
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data_60c
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data_60c

cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_120_{}.data_40c
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_100_{}.data_40c
cat ranges_helen.txt | parallel --verbose python model_test.py datasets/helen.data -m helen_80_{}.data_40c

# MUCT
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data --ismuct

cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data_80c --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data_80c --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data_80c --ismuct

cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data_60c --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data_60c --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data_60c --ismuct

cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_55_{}.data_40c --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_60_{}.data_40c --ismuct
cat ranges_muct.txt | parallel --verbose python model_test.py datasets/muct.data -m muct_65_{}.data_40c --ismuct
