import numpy as np
import random
import datetime
import networkx as nx
import math

path = '../data/'


drug_sim = np.loadtxt(path +"BioHNsdata/drug_sim.txt")
pro_sim = np.loadtxt(path +"BioHNsdata/protein_sim.txt")
disease_sim = np.loadtxt(path +"BioHNsdata/disease_sim.txt")
drug_num, pro_num, disease_num = drug_sim.shape[0], pro_sim.shape[0], disease_sim.shape[0]
total_num = drug_num + pro_num + disease_num 


def sim_contrast(pairnum):
    sample_list=[]
    count=0
    for i in range(total_num):
        if i<721:
            matsim=drug_sim
            numsize=drug_num
            k=0
        elif i>720 and i< 2615:
            matsim=pro_sim
            numsize=pro_num
            k=drug_num
        else:
            matsim=disease_sim
            numsize=disease_num
            k=drug_num+pro_num
           
        for j in range (pairnum):   #  a certain number of three tuples
            rand1=np.random.randint(numsize)
            rand2=np.random.randint(numsize)
            diff= matsim[i-k,rand1]-matsim[i-k,rand2]
            while diff<0:
                rand1=np.random.randint(numsize)
                rand2=np.random.randint(numsize)                
                diff=matsim[i-k,rand1]-matsim[i-k,rand2]
            sample_list.append([i,rand1+k,rand2+k,diff])
            count+=1
    simsample=open(path+"SSLdata/SimCon.txt", 'w')
    for i in range(len(sample_list)):
        simsample.write(str(sample_list[i][0])+ " "+ str(sample_list[i][1])+ " "+ str(sample_list[i][2])+" "+ str(sample_list[i][3]) + "\n")
    print("sample number:",count)
            

            
if __name__ == "__main__":
    sim_contrast(pairnum=15)

