import numpy as np
import random
import datetime
import networkx as nx
import math

path = '../data/'

def Biograph():
    BioHNs = np.loadtxt(path + "BioHNsdata/BioHNs.txt", dtype=int)
    G=nx.Graph()
    for i in range(BioHNs.shape[0]):
            for j in range(BioHNs.shape[0]):
                if BioHNs[i,j]==1:
                    G.add_edge(i,j)
    node_num, edge_num = len(G.nodes), len(G.edges)
    print('number of nodes and edges:', node_num, edge_num)
    return G, BioHNs

    
def clustering_coefficient():
    G, BioHNs=Biograph()
    node_cent=open(path + "SSLdata/ClusterPre.txt",'w')
    for i in range(len(G.nodes)):
        node_cent.write(str(i) + " " + str(nx.clustering(G,i)) + "\n")

    



if __name__ == "__main__":
    clustering_coefficient()


    