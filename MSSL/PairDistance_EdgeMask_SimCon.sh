#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python models/train_class.py --train_file ../data/SSLdata/PairDistance_train.txt --test_file ../data/SSLdata/PairDistance_test.txt --save PairDistance_EdgeMask_SimCon --batch_size 128 --length 2 --nclass 4 --sub 1 --ntask 3 --task 0 --epochs 1 --lr 5e-4 > PairDistance_EdgeMask_SimCon.log
CUDA_VISIBLE_DEVICES=0 python models/train_class.py --train_file ../data/SSLdata/EdgeMask_train.txt --test_file ../data/SSLdata/EdgeMask_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --batch_size 128  --length 2 --nclass 5 --ntask 3 --task 1 --epochs 1 --lr 5e-4 >> PairDistance_EdgeMask_SimCon.log
CUDA_VISIBLE_DEVICES=0 python models/train_sim.py --train_file ../data/SSLdata/SimCon_train.txt --test_file ../data/SSLdata/SimCon_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --batch_size 128 --mode 1 --ntask 3 --task 2 --lr 5e-4  --epochs 1 >> PairDistance_EdgeMask_SimCon.log
for ((i=2; i<=5; i++))
do
rnd=$(($RANDOM%3))
case $rnd in
  0)
  CUDA_VISIBLE_DEVICES=0 python models/train_class.py --train_file ../data/SSLdata/PairDistance_train.txt --test_file ../data/SSLdata/PairDistance_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --refine PairDistance_EdgeMask_SimCon --batch_size 128 --length 2 --nclass 4 --sub 1 --ntask 3 --task 0 --time $i --epochs 1 --lr 5e-4 >> PairDistance_EdgeMask_SimCon.log
  ;;
  1)
  CUDA_VISIBLE_DEVICES=0 python models/train_class.py --train_file ../data/SSLdata/EdgeMask_train.txt --test_file ../data/SSLdata/EdgeMask_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --refine PairDistance_EdgeMask_SimCon --batch_size 128  --length 2 --nclass 5 --ntask 3 --task 1 --time $i --epochs 1 --lr 5e-5 >> PairDistance_EdgeMask_SimCon.log
  ;;
  2)
  CUDA_VISIBLE_DEVICES=0 python models/train_sim.py --train_file ../data/SSLdata/SimCon_train.txt --test_file ../data/SSLdata/SimCon_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --refine PairDistance_EdgeMask_SimCon --batch_size 128 --mode 1 --ntask 3 --task 2 --time $i --lr 5e-4 --epochs 1 >> PairDistance_EdgeMask_SimCon.log
 ;;
esac
done

CUDA_VISIBLE_DEVICES=0 python models/get_feature.py --model PairDistance_EdgeMask_SimCon --length 3
