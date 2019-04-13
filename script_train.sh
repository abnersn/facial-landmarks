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
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data -p 50 --range {} -o muct_50_{}.data
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data -p 40 --range {} -o muct_40_{}.data
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data -p 30 --range {} -o muct_30_{}.data

cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_80c -p 50 --range {} -o muct_50_{}.data_80c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_80c -p 40 --range {} -o muct_40_{}.data_80c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_80c -p 30 --range {} -o muct_30_{}.data_80c

cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_60c -p 50 --range {} -o muct_50_{}.data_60c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_60c -p 40 --range {} -o muct_40_{}.data_60c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_60c -p 30 --range {} -o muct_30_{}.data_60c

cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_40c -p 50 --range {} -o muct_50_{}.data_40c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_40c -p 40 --range {} -o muct_40_{}.data_40c
cat ranges_muct.txt | parallel --verbose python model_train.py datasets/muct.data_40c -p 30 --range {} -o muct_30_{}.data_40c