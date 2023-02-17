import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import os
import copy

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--input_file', type=str, default='./', help='the path of input data.')
parser.add_argument('--feature', type=str, default='./', help='the path of node representations.')
parser.add_argument('--batch_size', type=int, default=128, help='the value of batch size.')
parser.add_argument('--ratio', type=float, default=0.05, help='the ratio of test dataset')
parser.add_argument('--save', type=str, default='./', help='the path to save model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)


with open(args.input_file, "r") as f:
    input_data = f.readlines()
feature = torch.load(args.feature)
input_feature = feature.shape[1]*2
feature = feature.tolist()
print("load input successfully!")

def test():
    predicts, labels = [], []
    for X, y in test_iter:
        X = X.cuda()
        y_hat = model(X)
        predicts.extend(y_hat.view(-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    predicts, labels = np.array(predicts), np.array(labels)
    auc = roc_auc_score(labels, predicts)
    aupr = average_precision_score(labels, predicts)
    return round(auc, 3), round(aupr, 3)

    
def train():
    for X, y in train_iter:
        X = X.cuda()
        y = y.view(-1, 1).float().cuda()
        optimizer.zero_grad()
        y_hat = model(X)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()

print("Ratio of the test dataset: "+ str(args.ratio))
print("learning rate: "+ str(args.lr))
all_auc, all_aupr = [], []
t_total = time.time()
frequency = int(args.epochs / 5)

model = nn.Sequential(
    #     nn.Linear(input_feature, 1),
    nn.Linear(input_feature, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
loss = nn.BCELoss()
model.cuda()
best_auc, best_aupr = 0, 0
filename = args.save.split('/')

train_iter, test_iter = load_cold_data(input_data, feature, args.batch_size, filename[0])     
for i in range(args.epochs):
    train()
    auc, aupr = test()
    print("epoch: "+str(i)+" auc: "+str(auc)+" aupr: "+str(aupr))
    best_model = copy.deepcopy(model)
    best_auc = auc
    best_aupr = aupr

torch.save(best_model, args.save+'/best_model.pt')
print("AUC: ")
print(best_auc)
print("AUPR: ")
print(best_aupr)
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

