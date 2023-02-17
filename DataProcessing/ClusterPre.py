import numpy as np
import random
import datetime
import networkx as nx
import math
import argparse
from utils import Biograph

path = '../data/'
parser = argparse.ArgumentParser()
parser.add_argument('--downstream', type=str, default='DDI', help='The name of downstream')
parser.add_argument('--scenario', type=str, default='warm', help='The test scenario of downstream')
parser.add_argument('--dataclean', type=int, default=0, help='Whether to remove the test data from SSL dataset.')   #  

args = parser.parse_args()




def clustering_coefficient():
    G, BioHNs =Biograph(args.downstream, args.scenario, args.dataclean)
    node_cent=open(path + "SSLdata/ClusterPre.txt",'w')
    for i in range(len(G.nodes)):
        node_cent.write(str(i) + " " + str(nx.clustering(G,i)) + "\n")




if __name__ == "__main__":
    clustering_coefficient()

    # Biograph()

    
