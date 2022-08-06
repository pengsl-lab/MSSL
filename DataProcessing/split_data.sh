#!/bin/bash

python create_dataset.py --input_file ../data/SSLdata/ClusterPre.txt
python create_dataset.py --input_file ../data/SSLdata/PairDistance.txt
python create_dataset.py --input_file ../data/SSLdata/PathClass.txt
python create_dataset.py --input_file ../data/SSLdata/EdgeMask.txt
python create_dataset.py --input_file ../data/SSLdata/SimReg.txt
python create_dataset.py --input_file ../data/SSLdata/SimCon.txt