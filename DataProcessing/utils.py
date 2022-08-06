import numpy as np
import random
import datetime
import networkx as nx
import math

path = '../data/'



def Biograph(downstream, scenario, dataclean):
    BioHNs = np.loadtxt(path + "BioHNsdata/BioHNs.txt", dtype=int)


    G=nx.Graph()
    for i in range(BioHNs.shape[0]):
            for j in range(BioHNs.shape[0]):
                if BioHNs[i,j]==1:
                    G.add_edge(i,j)
    node_num, edge_num = len(G.nodes), len(G.edges)
    print('number of nodes and edges:', node_num, edge_num)

    if dataclean:
        test_index = np.loadtxt(path + "DownStreamdata/" + downstream + "net_" + scenario + "_test.txt", dtype=int)
        test_index = np.array(list(set([tuple(t) for t in test_index])))
        for edg in test_index:
            if edg[2] == 1:
                G.remove_edge(edg[0], edg[1])
                BioHNs[edg[0], edg[1]] = 0
                BioHNs[edg[1], edg[0]] = 0
        node_num, edge_num = len(G.nodes), len(G.edges)
        print('number of clean nodes and edges:', node_num, edge_num)
    return G, BioHNs