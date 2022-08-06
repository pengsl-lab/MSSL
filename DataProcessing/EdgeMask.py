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

def mask_edge():
    G, BioHNs =Biograph(args.downstream, args.scenario, args.dataclean)
    edge, false=[], []
    c1, c2, c3, c4 = 0, 0, 0, 0
    for eg in G.edges():
        eglist=list(eg)
        eglist.sort()
        neg=list(G.neighbors(eglist[0]))
        if eglist[0]<721:
            if eglist[1]<721:
                for i in range(1):
                    # if np.random.rand()<0.5:   #  a certain number of drug-drug interaction
                    edge.append([eglist[0], eglist[1], 1])
                    rand=np.random.randint(721)
                    while(rand in neg):
                        rand=np.random.randint(721)
                    false.append([eglist[0], rand, 4])
                    c1 += 1
            elif eglist[1]<2615:
                for i in range(1):   #  a certain number of drug-protein interaction
                    edge.append([eglist[0], eglist[1], 2])
                    rand=np.random.randint(721, 2615)
                    while(rand in neg):
                        rand=np.random.randint(721, 2615)
                    false.append([eglist[0], rand, 4])
                    c2 += 1
            else:
                for i in range(1):   #  a certain number of drug-disease interaction
                    edge.append([eglist[0], eglist[1], 3])
                    rand=np.random.randint(2615, 3046)
                    while(rand in neg):
                        rand=np.random.randint(2615, 3046)
                    false.append([eglist[0], rand, 4])
                    c3 += 1
        else:
            if eglist[1]<2615:
                for i in range(1):  #  a certain number of protein-protein interaction
                    edge.append([eglist[0], eglist[1], 1])
                    rand=np.random.randint(721, 2615)
                    while(rand in neg):
                        rand=np.random.randint(721, 2615)
                    false.append([eglist[0], rand, 4])
                    c1 += 1
            else:
                for i in range(1):     #  a certain number of protein-disaese interaction
                    edge.append([eglist[0], eglist[1], 0])
                    rand=np.random.randint(2615, 3046)
                    while(rand in neg):
                        rand=np.random.randint(2615, 3046)
                    false.append([eglist[0], rand, 4])
                    c4 += 1
    temp = np.array(edge)
    # temp = np.vstack((np.array(edge), np.array(false)))
    np.savetxt(path + "/SSLdata/EdgeMask.txt", temp, fmt="%d")
    print('the number of each class:', c1, c2, c3, c4, c1+c2+c3+c4)

if __name__ == "__main__":
    mask_edge()

