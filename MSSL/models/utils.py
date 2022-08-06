import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as Data

def load_graph(feature_file='../data/BioHNsdata/BioHNs_code.txt', adj_file='../data/BioHNsdata/BioHNs_clean.txt'):
    with open(feature_file, 'r') as f:
        lines = f.readlines()
    features = []
    for line in lines:
        line = line.split(' ')
        line = [int(x) for x in line]
        features.append(line)
    features = torch.tensor(features, dtype=torch.float)
    adj = []
    with open(adj_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        line = [int(x) for x in line]
        adj.append(line)
    adj = torch.tensor(adj, dtype=torch.float)
    return features, adj

def load_dataset(data_file, batch_size, mode=0, shuffle=True, sub=0):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    data_X, data_y = [], []
    for line in lines:
        X, y = [], []
        line = line.split(' ')
        X, y = line[:-1], line[-1]
        X = [int(x) for x in X]
        y = float(y) if mode else int(y)
        if sub:
            y = y - 1
        data_X.append(X)
        data_y.append(y)
    data_X = torch.tensor(data_X, dtype=torch.long)
    if mode:
        data_y = torch.tensor(data_y, dtype=torch.float)
    else:
        data_y = torch.tensor(data_y, dtype=torch.long)
    data_dataset = Data.TensorDataset(data_X, data_y)
    data_iter = Data.DataLoader(data_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_iter
    
