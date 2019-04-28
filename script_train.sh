#!/bin/bash

# HELEN
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data -p 120 --range {} -o helen_120_{}.data
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data -p 100 --range {} -o helen_100_{}.data
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data -p 80 --range {} -o helen_80_{}.data

cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_80c -p 120 --range {} -o helen_120_{}.data_80c
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_80c -p 100 --range {} -o helen_100_{}.data_80c
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_80c -p 80 --range {} -o helen_80_{}.data_80c

cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_60c -p 120 --range {} -o helen_120_{}.data_60c
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_60c -p 100 --range {} -o helen_100_{}.data_60c
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_60c -p 80 --range {} -o helen_80_{}.data_60c

cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_40c -p 120 --range {} -o helen_120_{}.data_40c
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_40c -p 100 --range {} -o helen_100_{}.data_40c
cat ranges_helen.txt | parallel --verbose python model_train.py datasets/helen.data_40c -p 80 --range {} -o helen_80_{}.data_40c

# MUCT
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data -p 55 --range {} -o muct_55_{}.data
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data -p 60 --range {} -o muct_60_{}.data
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data -p 65 --range {} -o muct_65_{}.data

cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_80c -p 55 --range {} -o muct_55_{}.data_80c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_80c -p 60 --range {} -o muct_60_{}.data_80c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_80c -p 65 --range {} -o muct_65_{}.data_80c

cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_60c -p 55 --range {} -o muct_55_{}.data_60c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_60c -p 60 --range {} -o muct_60_{}.data_60c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_60c -p 65 --range {} -o muct_65_{}.data_60c

cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_40c -p 55 --range {} -o muct_55_{}.data_40c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_40c -p 60 --range {} -o muct_60_{}.data_40c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_40c -p 65 --range {} -o muct_65_{}.data_40c
