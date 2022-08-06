import numpy as np
import random
import datetime
import networkx as nx
import os
import shutil
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


def cur_neighbor(G, cur_node, start, end):
    neighbor = list(G.neighbors(cur_node))
    neighbors = []
    for tem in neighbor:
        if tem>=start and tem<end:
            neighbors.append(tem)
    return neighbors     


def writepath(walk, string, label):
    filename = path + "SSLdata/metapath/"+string+".txt"
    randpath = open(filename, 'a')
    
    length = len(walk)
    for i in range(length):
        if i==length-1:
            randpath.write(str(walk[i]) + " " + label + "\n")
        else:
            randpath.write(str(walk[i]) + " " )


def Mwalk(start_node, G, BioHNs, num, string, label):
    walk = [start_node]
    walk_length = len(num)
    for i in range(walk_length):
        cur_nbrs = cur_neighbor(G, walk[-1], num[i][0], num[i][1])
        if len(cur_nbrs)==0:
            return
        walk.append(random.choice(cur_nbrs))
        
    if np.random.randint(1,3)==1:
        walk=list(reversed(walk))
        label=label+1
        string=string[::-1]
        
    writepath(walk, string, str(label))
    if np.random.randint(1,16)==1:
        falsepath(G, BioHNs, walk, walk_length)
    return


# 721  (0,721)
# 1894  (721, 2615)
# 431 (2615, 3046)


def falsepath(G, BioHNs, walk, walk_length):
    while(1):
        copy_walk = walk
        randlist = list(set(np.random.randint(0,walk_length, size=np.random.randint(1,walk_length))))
        for rand in randlist:
            if copy_walk[rand] < 721:
                nod = np.random.randint(0, 721)
            elif copy_walk[rand] < 2615:
                nod = np.random.randint(721, 2615)
            else:
                nod = np.random.randint(2615, 3046)

            copy_walk[rand] = nod
        sum = 0
        for i in range(walk_length-1):
            sum = sum + BioHNs[copy_walk[i], copy_walk[i+1]]
            
        num=0
        if sum < walk_length:
            writepath(copy_walk, "falsepath", '0')
            break
            


def simulate_walks(num_walks):
    DDDT = [[0, 721],[0, 721],[721, 2615]]
    DDTT = [[0, 721],[721, 2615],[721, 2615]]
    DDST = [[0, 721],[2615, 3046],[721, 2615]]
    DTDT = [[721, 2615],[0, 721],[721, 2615]]
    DTTT = [[721, 2615],[721, 2615],[721, 2615]]
    DTST = [[721, 2615],[2615, 3046],[721, 2615]]
    DSDT = [[2615, 3046],[0, 721],[721, 2615]]
    DSTT = [[2615, 3046],[721, 2615],[721, 2615]]

    G, BioHNs =Biograph(args.downstream, args.scenario, args.dataclean)
    node = list(G.nodes())
    nodes = []
    for nod in node:
        if nod<721:
            nodes.append(nod)
            
    print('Begin sample of meta path...')
    for walk_iter in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:   
            Mwalk(node, G, BioHNs, DDDT, "DDDT", 1)
            Mwalk(node, G, BioHNs, DDTT, "DDTT", 3)
            Mwalk(node, G, BioHNs, DDST, "DDST", 5)
            Mwalk(node, G, BioHNs, DTDT, "DTDT", 7)
            Mwalk(node, G, BioHNs, DTTT, "DTTT", 9)
            Mwalk(node, G, BioHNs, DTST, "DTST", 11)
            Mwalk(node, G, BioHNs, DSDT, "DSDT", 13)
            Mwalk(node, G, BioHNs, DSTT, "DSTT", 15)

            
def allmetapath(num_walks):
    if os.path.exists(path+'SSLdata/metapath')==0:
        os.makedirs(path+'SSLdata/metapath')
    simulate_walks(num_walks)
    sum = 0
    subpath = ["DDDT", "DDTT", "DDST", "DTDT", "DTTT", "DTST", "DSDT", "DSTT", "TDDD", "TDTD", "TDSD", "TTDD", "TTTD", "TTSD", "TSDD", "TSTD", "falsepath"]

    metapath = open(path+"SSLdata/PathClass.txt", "w")
    sum = 0
    for pathtype in subpath:
        num = 0
        pathline = open(path+"SSLdata/metapath/" + pathtype + ".txt", "r")
        subpath = pathline.readline()

        while (subpath):
            metapath.write(subpath)
            num += 1
            sum += 1
            subpath = pathline.readline()    
        print(pathtype, num)
    print("path total number", sum)
    shutil.rmtree(path+'SSLdata/metapath')


if __name__ == "__main__":
    allmetapath(num_walks=25)   # a certain number of node pairs

