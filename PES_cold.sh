#!/bin/bash

cd DownStream
python create_sample.py --input ../data/DownStreamdata/DTInet.txt --offset 1

cd ../DataProcessing

python degree_code.py --downstream DTI   --scenario cold --dataclean 1 >logpre
python PairDistance.py  --downstream DTI   --scenario cold --dataclean 1 >>logpre
python SimCon.py  --downstream DTI   --scenario cold  --dataclean 1 >>logpre
python EdgeMask.py  --downstream DTI   --scenario cold --dataclean 1 >>logpre
# python PathClass.py  --downstream DTI   --scenario cold --dataclean 1 >>logpre
# python SimReg.py  --downstream DTI   --scenario cold  --dataclean 1 >>logpre
# python ClusterPre.py   --downstream DTI   --scenario cold --dataclean 1 >>logpre


python create_dataset.py --input_file ../data/SSLdata/PairDistance.txt
python create_dataset.py --input_file ../data/SSLdata/EdgeMask.txt
python create_dataset.py --input_file ../data/SSLdata/SimCon.txt
# python create_dataset.py --input_file ../data/SSLdata/SimReg.txt
# python create_dataset.py --input_file ../data/SSLdata/PathClass.txt
# python create_dataset.py --input_file ../data/SSLdata/ClusterPre.txt

cd ../MSSL

python models/train_class.py --train_file ../data/SSLdata/PairDistance_train.txt --test_file ../data/SSLdata/PairDistance_test.txt --save PairDistance_EdgeMask_SimCon --batch_size 128 --length 2 --nclass 4 --sub 1 --ntask 3 --task 0 --epochs 1 --lr 5e-4 > PairDistance_EdgeMask_SimCon.log
python models/train_class.py --train_file ../data/SSLdata/EdgeMask_train.txt --test_file ../data/SSLdata/EdgeMask_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --batch_size 128  --length 2 --nclass 4 --ntask 3 --task 1 --epochs 1 --lr 5e-4 >> PairDistance_EdgeMask_SimCon.log
python models/train_sim.py --train_file ../data/SSLdata/SimCon_train.txt --test_file ../data/SSLdata/SimCon_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --batch_size 128 --mode 1 --ntask 3 --task 2 --lr 5e-4  --epochs 1 >> PairDistance_EdgeMask_SimCon.log
for ((i=1; i<=30; i++))
do
rnd=$(($RANDOM%3))
case $rnd in
  0)
 python models/train_class.py --train_file ../data/SSLdata/PairDistance_train.txt --test_file ../data/SSLdata/PairDistance_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --refine PairDistance_EdgeMask_SimCon --batch_size 128 --length 2 --nclass 4 --sub 1 --ntask 3 --task 0 --time $i --epochs 1 --lr 5e-4 >> PairDistance_EdgeMask_SimCon.log
  ;;
  1)
 python models/train_class.py --train_file ../data/SSLdata/EdgeMask_train.txt --test_file ../data/SSLdata/EdgeMask_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --refine PairDistance_EdgeMask_SimCon --batch_size 128  --length 2 --nclass 4 --ntask 3 --task 1 --time $i --epochs 1 --lr 5e-4 >> PairDistance_EdgeMask_SimCon.log
  ;;
  2)
 python models/train_sim.py --train_file ../data/SSLdata/SimCon_train.txt --test_file ../data/SSLdata/SimCon_test.txt --save PairDistance_EdgeMask_SimCon --share PairDistance_EdgeMask_SimCon --refine PairDistance_EdgeMask_SimCon --batch_size 128 --mode 1 --ntask 3 --task 2 --time $i --lr 5e-4 --epochs 1 >> PairDistance_EdgeMask_SimCon.log
 ;;
esac
done
#
###
python models/get_feature.py --model PairDistance_EdgeMask_SimCon --length 3


cd ../DownStream

python cold_start.py --input_file ../data/DownStreamdata/DTInet_sample.txt --feature ../MSSL/feature_PairDistance_EdgeMask_SimCon.pt --lr 0.002 --epochs 30 --save DTInet/PairDistance_EdgeMask_SimCon_cold >> DTI_PairDistance_EdgeMask_SimCon



