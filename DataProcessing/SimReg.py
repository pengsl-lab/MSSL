import numpy as np
import random
import datetime
import networkx as nx
import math

path = '../data/'

drug_sim = np.loadtxt(path+"BioHNsdata/drug_sim.txt")
pro_sim = np.loadtxt(path+"BioHNsdata/protein_sim.txt")
disease_sim = np.loadtxt(path+"BioHNsdata/disease_sim.txt")
drug_num, pro_num, disease_num = drug_sim.shape[0], pro_sim.shape[0], disease_sim.shape[0]


def sim_regression(dgpair=70, propair=180, dispair=40):  # a certain number of node pairs
    simsample=open(path+"SSLdata/SimReg.txt", 'w')
    kdg, kpro, kdis = 0, 0, 0
    for i in range(drug_num):
        dg=np.random.choice(drug_num, dgpair, replace=False)   #  a certain number of drug pairs
        for j in dg:
            simsample.write(str(i)+ " "+ str(j)+ " "+ str(drug_sim[i][j])+ "\n")
            kdg += 1
            
    for i in range(pro_num):
        pro=np.random.choice(pro_num, propair, replace=False)   #  a certain number of protein pairs
        for j in pro:
            simsample.write(str(i+drug_num)+ " "+ str(j+drug_num)+ " "+ str(pro_sim[i][j])+ "\n")
            kpro += 1
            
    for i in range(disease_num):
        dis=np.random.choice(disease_num, dispair, replace=False)   #  a certain number of protein pairs
        for j in dis:
            simsample.write(str(i+drug_num+pro_num)+ " "+ str(j+drug_num+pro_num)+ " "+ str(disease_sim[i][j])+ "\n")
            kdis += 1
    print("similirty of drug, protein and disease: ", kdg, kpro, kdis, kdg+kpro+kdis)
            
if __name__ == "__main__":
    sim_regression(dgpair=18, propair=48, dispair=12)   # a certain number of node pairs

