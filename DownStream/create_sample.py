import argparse
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./', help='path of the input dataset.')
parser.add_argument('--offset', type=int, default=0, help='the choice of downstream tasks. 0 is DDI, 1 is DTI')
parser.add_argument('--seed', type=int, default=18, help='Random seed.')  
parser.add_argument('--warm_ratio', type=float, default=0.1, help='the ratio of test dataset in warm start prediction')
parser.add_argument('--cold_ratio', type=float, default=0.05, help='the ratio of test dataset in warm start prediction')

args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = f.readlines()
num_postive, num_negtive = 0, 0
for line in lines:
    line = line.split(' ')
    for v in line:
        if int(v) == 0:
            num_negtive += 1
        else :
            num_postive += 1

ratio = num_postive / num_negtive
num_postive, num_negtive = 0, 0
random.seed(args.seed)   
num_sample = []
num_row = len(lines)
num_drug, num_pro = 721, 1894


if args.offset:
    offset = num_drug
else:
    offset = 0

for i in range(num_row):
    line = lines[i].split(' ')
#     num_column = len(line)
    if args.offset:
        num = len(line)
    else:
        num = i

    for j in range(num):
        v = int(line[j])
        if v == 1:
            num_postive += 1
            num_sample.append(str(i) + ' ' + str(j+offset) + ' ' + str(v))
        else:
            if random.random()<ratio:
                num_negtive += 1
                num_sample.append(str(i) + ' ' + str(j+offset) + ' ' + str(v))
print("Number of positive: "+str(num_postive))
print("Number of negtive: "+str(num_negtive))
print("Total: "+str(len(num_sample)))


BioHNs = np.loadtxt("../data/BioHNsdata/BioHNs.txt", dtype=int)
isolated_node = []
for i in range(num_drug):
    if args.offset:
        if np.sum(BioHNs[i])-np.sum(BioHNs[i, num_drug:num_drug+num_pro]) ==0:
            isolated_node.append(i)
    else:
        if np.sum(BioHNs[i, num_drug:])==0:
            isolated_node.append(i)

random.shuffle(num_sample)
warm_num_test = int(len(num_sample)*args.warm_ratio)
warm_test_sample = []
for i in range(warm_num_test):
    while (1):
        rad = random.randint(0, len(num_sample)-1)
        sample = num_sample[rad].split(' ')
        if int(sample[0]) not in isolated_node and int(sample[1]) not in isolated_node and int(sample[0]) not in warm_test_sample and int(sample[1]) not in warm_test_sample:
            break
    warm_test_sample.append(rad)
np.savetxt(args.input[:args.input.rfind('.')] + '_warm_node.txt', warm_test_sample, fmt='%d')

warm_test_set = args.input[:args.input.rfind('.')] + '_warm_test.txt'
with open(warm_test_set, 'w') as f:
    for i in warm_test_sample:
        f.write(num_sample[i] +'\n')

cold_num_test = int(num_drug*args.cold_ratio)
cold_nodes = []
for i in range(cold_num_test):
    while(1):
        rad = random.randint(0, num_drug-1)
        if rad not in isolated_node and rad not in cold_nodes:
            break
    cold_nodes.append(rad)
np.savetxt(args.input[:args.input.rfind('.')] + '_cold_node.txt', cold_nodes, fmt='%d')


cold_test_set = args.input[:args.input.rfind('.')] + '_cold_test.txt'
with open(cold_test_set, 'w') as f:
    for i in range(len(num_sample)):
        sample = num_sample[i].split(' ')
        for j in cold_nodes:
            if j == int(sample[0]) or j == int(sample[1]):
                f.write(num_sample[i] +'\n')


save_name = args.input[:args.input.rfind('.')] + '_sample.txt'
with open(save_name, 'w') as f:
    for sample in num_sample:
        f.write(sample+'\n')
print("Write to file successfully: "+save_name)
