# MSSL2drug
We peopose multi-task joint strategies of self-supervised representation learning on biomedical networks for drug discovery, named MSSL2drug. We design six basic SSL tasks that are inspired by various modality features including structures, semantics, and attributes in biomedical heterogeneous networks. In addition, fifteen combinations of multiple tasks are evaluated by a graph attention-based adversarial multi-task learning framework in two drug discovery scenarios. The results suggest two important findings. (1) Combinations of multimodal tasks achieve the best performance compared to other multi-task joint strategies. (2) The joint training of local and global SSL tasks yields higher performance than random task combinations.

## Data description
1. BioHNsdata: biomedical heterogeneous networks
* drug.txt: list of drug.
* protein.txt: list of protein.
* disease.txt: list of disease.
* drug_sim.txt: drug similarity matrix
* protein_sim.txt: protein similarity matrix
* disease_sim.txt: protein similarity matrix
* BioHNs.txt: the adjacency matrix of the biomedical heterogeneous networks.
* BioHNs_code.txt: the initialization features of nodes in the biomedical heterogeneous network.
* BioHNsdata_with order: the corresponding order of node in drug_sim.txt, protein_sim.txt, disease_sim.txt, and BioHNs.txt BioHNs.txt, and BioHNs_code.txt
2. DownStreamdata: drug discovery data
* DDINet.txt: Drug-Drug interaction matrix.
* DTINet.txt: Drug-Protein interaction matrix.
3. SSLdata: Self-supervised representation pretext
* ClusterPre.txt, PairDistance.txt, EdgeMask.txt, PathClass.txt, SimReg.txt and SimCon.txt: a example dataset for six self-supervised task.
* SSL_data_description.txt : The detailed description of PairDistance.txt, EdgeMask.txt, PathClass.txt, SimReg.txt and SimCon.txt

## Quick start

Run NeoDTI_cv.py to reproduce the cross validation results of NeoDTI.

* bash generating_SSL.sh: generating the training labels for six SSL pretext tasks.
* bash split_data.sh: spliting dataset to training and test set for six self-supervised learning.

## Single task-driven self-supervised representation learning
[Graph attention networks (GATs)](https://arxiv.org/abs/1710.10903v3) are used to train the single task-driven self-supervised representation learning. The code of GAT can be downloaded from https://github.com/Diego999/pyGAT.

## Multi-task self-supervised representation learning
Switch to the MSSL directory

`models/` contains the different implementations of graph attention-based adversarial multi-task learning models for three different training paradigm, including classification task, regression task and similarity constrast task. In MSSL2drug, we develop fifteen combinations of multiple tasks. However, there are same implementations among some multi-task combinations. Therefore, we are the following examples:

* bash PairDistance_EdgeMask_SimCon.sh: Get the embedding based on PairDistance-EdgeMask-SimCon:


## DDI and DTI predictions
Switch to the DownStream directory
### Create downstream dataset

--offset 0 is DDI network and 1 is DTI network

`python create_sample.py --input ../data/DownStreamdata/DDInet.txt --offset 0`

`python create_sample.py --input ../data/DownStreamdata/DTInet.txt --offset 1`

### DDI predictions in warm-start scenarios  
`python warm_start.py --input_file ../data/DownStreamdata/DDInet_sample.txt --feature ../MSSL/feature_PairDistance_EdgeMask_SimCon.pt --lr 0.002 --epochs 30 --save DDInet/PairDistance_EdgeMask_SimCon`

### DTI predictions in warm-start scenarios  
`python warm_start.py --input_file ../data/DownStreamdata/DTInet_sample.txt --feature ../MSSL/feature_PairDistance_EdgeMask_SimCon.pt --lr 0.002 --epochs 30 --save DTInet/PairDistance_EdgeMask_SimCon`

### DDI predictions in cold-start scenarios  
`python cold_start.py --input_file ../data/DownStreamdata/DDInet_sample.txt --feature ../MSSL/feature_PairDistance_EdgeMask_SimCon.pt --lr 0.002 --epochs 30 --save cold_DDInet/PairDistance_EdgeMask_SimCon`

### DTI predictions in cold-start scenarios  
`python cold_start.py --input_file ../data/DownStreamdata/DTInet_sample.txt --feature ../MSSL/feature_PairDistance_EdgeMask_SimCon.pt --lr 0.002 --epochs 30 --save cold_DTInet/PairDistance_EdgeMask_SimCon`

## Requirements
MSSL2drug is tested to work under:
* Python 3.7.7
* numpy 1.16.1
* torch 1.6.0

# Contacts
If you have any questions or comments, please feel free to email: xqw@hnu.edu.cn or jicheng@hnu.edu.cn.
