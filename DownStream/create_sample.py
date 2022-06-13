import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./', help='path of the input dataset.')
parser.add_argument('--offset', type=int, default=0, help='0 is DDI, 1 is DTI')

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
num_sample = []
num_row = len(lines)
if args.offset:
    offset = 721
else:
    offset = 0
for i in range(num_row):
    line = lines[i].split(' ')
#     num_column = len(line)
    for j in range(i):
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
save_name = args.input[:args.input.rfind('.')] + '_sample.txt'
with open(save_name, 'w') as f:
    for sample in num_sample:
        f.write(sample+'\n')
print("Write to file successfully: "+save_name)
