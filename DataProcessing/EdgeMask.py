import numpy as np
import random
import datetime
import networkx as nx
import os
from ClusterPre import Biograph

# 721  (0,721)
# 1894  (721, 2615)
# 431 (2615, 3046)

path = '../data/'

def mask_edge():
    G, BioHNs =Biograph()
    edge, false=[], []
    c1, c2, c3, c4 = 0, 0, 0, 0
    for eg in G.edges():
        eglist=list(eg)
        eglist.sort()
        neg=list(G.neighbors(eglist[0]))
        if eglist[0]<721:
            if eglist[1]<721:
                if np.random.rand()<0.017:   #  a certain number of drug-drug interaction
                    edge.append([eglist[0], eglist[1], 1])
                    rand=np.random.randint(721)
                    while(rand not in neg):
                        rand=np.random.randint(721)
                    false.append([eglist[0], rand, 0])
                    c1 += 1
            elif eglist[1]<2615:
                for i in range(2):   #  a certain number of drug-protein interaction
                    edge.append([eglist[0], eglist[1], 2])
                    rand=np.random.randint(721, 2615)
                    while(rand not in neg):
                        rand=np.random.randint(721, 2615)
                    false.append([eglist[0], rand, 0])
                    c2 += 1
            else:
                for i in range(2):   #  a certain number of drug-disease interaction
                    edge.append([eglist[0], eglist[1], 3])
                    rand=np.random.randint(2615, 3046)
                    while(rand not in neg):
                        rand=np.random.randint(2615, 3046)
                    false.append([eglist[0], rand, 0])
                    c3 += 1
        else:
            if eglist[1]<2615:
                if np.random.randint(16)<2:  #  a certain number of protein-protein interaction
                    edge.append([eglist[0], eglist[1], 1])
                    rand=np.random.randint(721, 2615)
                    while(rand not in neg):
                        rand=np.random.randint(721, 2615)
                    false.append([eglist[0], rand, 0])
                    c1 += 1
            else:
                edge.append([eglist[0], eglist[1], 4])  #  a certain number of protein-disaese interaction
                rand=np.random.randint(2615, 3046)
                while(rand not in neg):
                    rand=np.random.randint(2615, 3046)
                false.append([eglist[0], rand, 0])
                c4 += 1
    temp = np.array(edge)
    # temp = np.vstack((np.array(edge), np.array(false)))
    np.savetxt(path + "/SSLdata/EdgeMask.txt", temp, fmt="%d")
    print('the number of each class:', c1, c2, c3, c4, c1+c2+c3+c4)

if __name__ == "__main__":
    mask_edge()

