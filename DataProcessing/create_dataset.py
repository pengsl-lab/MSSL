import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default=False, help='input file.')

args = parser.parse_args()

with open(args.input_file, 'r') as f:
    lines = f.readlines()

random.shuffle(lines)
random.seed(1)
num_sample = len(lines)
num_test = int(num_sample*0.1)
test = lines[:num_test]
train = lines[num_test:]
file_prefix = args.input_file[:args.input_file.rfind('.')]
train_file = file_prefix + '_train.txt'
test_file = file_prefix + '_test.txt'

print("input file "+str(args.input_file))
with open(train_file, 'w') as f:
    f.writelines(train)
print("write training dataset successfully, the number of samples: "+ str(len(train)))
with open(test_file, 'w') as f:
    f.writelines(test)
print("write testing dataset successfully, the number of samples: "+ str(len(test)))