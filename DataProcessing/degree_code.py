import numpy as np
import random
import datetime
import networkx as nx
import os
from utils import Biograph
import argparse

# 721  (0,721)
# 1894  (721, 2615)
# 431 (2615, 3046)

path = '../data/'
parser = argparse.ArgumentParser()
parser.add_argument('--downstream', type=str, default='DDI', help='The name of downstream')
parser.add_argument('--scenario', type=str, default='warm', help='The test scenario of downstream')
parser.add_argument('--dataclean', type=bool, default=0, help='Whether to remove the test data from SSL dataset.')   #  

args = parser.parse_args()


def degree_code():
    G, BioHNs =Biograph(args.downstream, args.scenario, args.dataclean)
    drug_num, protein_num, disease_num = 721, 1894, 431

    drug, protein, disease = [], [], []
    nodes = sorted(list(G.nodes()))

    for i in nodes:
        neigh = list(G.neighbors(i))
        dg, pro, dis = 0, 0, 0
        for nod in neigh:
            if nod < drug_num:
                dg += 1
            elif nod >= drug_num and nod < drug_num + protein_num:
                pro += 1
            else:
                dis += 1
        if dg+pro+dis==0:
            if np.random.rand()<0.333:
                dg = 1
            elif np.random.rand()>=0.333 and np.random.rand()<0.666:
                pro = 1
            else:
                dis =1
        drug.append(dg)
        protein.append(pro)
        disease.append(dis)

    dg_col, pro_col, dis_col = len(list(set(drug))), len(list(set(protein))), len(list(set(disease)))
    dg_code, pro_code, dis_code = np.zeros((drug_num, dg_col + pro_col + dis_col)), np.zeros(
        (protein_num, dg_col + pro_col + dis_col)), np.zeros((disease_num, dg_col + pro_col + dis_col))
    dglist, prolist, dislist = sorted(list(set(drug))), sorted(list(set(protein))), sorted(list(set(disease)))

    for i in range(drug_num):
        dg_code[i, dglist.index(drug[i])] = 1
        dg_code[i, (prolist.index(protein[i])) + dg_col] = 1
        dg_code[i, (dislist.index(disease[i])) + dg_col + pro_col] = 1

    for i in range(protein_num):
        pro_code[i, dglist.index(drug[i + drug_num])] = 1
        pro_code[i, (prolist.index(protein[i + drug_num])) + dg_col] = 1
        pro_code[i, (dislist.index(disease[i + drug_num])) + dg_col + pro_col] = 1

    for i in range(disease_num):
        dis_code[i, dglist.index(drug[i + drug_num + protein_num])] = 1
        dis_code[i, (prolist.index(protein[i + drug_num + protein_num])) + dg_col] = 1
        dis_code[i, (dislist.index(disease[i + drug_num + protein_num])) + dg_col + pro_col] = 1

    BioHNs_code = np.vstack((dg_code, pro_code, dis_code))
    np.savetxt("../data/BioHNsdata/BioHNs_code.txt", BioHNs_code, fmt="%d")
    np.savetxt("../data/BioHNsdata/BioHNs_clean.txt", BioHNs, fmt="%d")

    print("dg_col, pro_col, dis_col", dg_col, pro_col, dis_col, BioHNs_code.shape)
    print("The number", dglist.index(drug[i]))


if __name__ == "__main__":
    degree_code()

