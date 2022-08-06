import numpy as np
import random
import datetime
import networkx as nx
from utils import Biograph
import argparse

path = '../data/'
parser = argparse.ArgumentParser()
parser.add_argument('--downstream', type=str, default='DDI', help='The name of downstream')
parser.add_argument('--scenario', type=str, default='warm', help='The test scenario of downstream')
parser.add_argument('--dataclean', type=bool, default=0, help='Whether to remove the test data from SSL dataset.')   #  

args = parser.parse_args()


def global_disdance():
    G, BioHNs =Biograph(args.downstream, args.scenario, args.dataclean)
    dis_sample=[]
    one_hop, two_hop, three_hop, four_hop = 0, 0, 0, 0  # 221140 3359298 5042652 597242
    prob = 0.06   # Sampling probability
    for i in range(len(G.nodes)):
        for u in range(len(G.nodes)):
            if nx.has_path(G, i, u):
                length=nx.shortest_path_length(G, i, u)
                if length==1 and np.random.rand(1)< prob:
                    dis_sample.append([i,u, length])
                    one_hop+=1
                elif length==2 and np.random.rand(1)< prob * 0.0658:
                    dis_sample.append([i,u, length])
                    two_hop+=1
                elif length==3 and np.random.rand(1)< prob * 0.0439:
                    dis_sample.append([i,u, length])
                    three_hop+=1
                elif length>3 and np.random.rand(1)< prob*0.3703:
                    dis_sample.append([i,u, 4])
                    four_hop+=1

    np.savetxt(path + "SSLdata/PairDistance.txt", np.array(dis_sample), fmt="%d")
    print("one_hop, two_hop, three_hop, four_hop", one_hop, two_hop, three_hop, four_hop, one_hop+two_hop+three_hop+four_hop)


if __name__ == "__main__":
    global_disdance()


