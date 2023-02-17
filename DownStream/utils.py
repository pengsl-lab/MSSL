import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
import random
import numpy as np

def load_dataset(input_file, feature_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
    feature = torch.load(feature_file)
    feature = feature.tolist()
    samples_X, samples_y = [], []
    for line in lines:
        line = line.split(' ')
        y = int(line[-1])
        samples_y.append(y)
        x = []
        for node in line[:-1]:
            x.extend(feature[int(node)])
        samples_X.append(x)
    samples_X = torch.tensor(samples_X, dtype=torch.float)
    samples_y = torch.tensor(samples_y, dtype=torch.long)
    input_feature = samples_X.shape[1]
    return input_feature, samples_X, samples_y

# def load_fold_iter(samples_X, samples_y, k, batch_size):
#     num_test = int(len(samples_X)*0.1)
#     if k == 0:
#         test_X, test_y = samples_X[:num_test], samples_y[:num_test]
#         train_X, train_y = samples_X[num_test:], samples_y[num_test:]
#     elif k == 9:
#         test_X, test_y = samples_X[num_test*k:], samples_y[num_test*k:]
#         train_X, train_y = samples_X[:num_test*k], samples_y[:num_test*k]
#     else:
#         test_X, test_y = samples_X[num_test*k:num_test*(k+1)], samples_y[num_test*k:num_test*(k+1)]
#         train_X = torch.cat([samples_X[0:num_test*k], samples_X[num_test*(k+1):]], dim=0)
#         train_y = torch.cat([samples_y[0:num_test*k], samples_y[num_test*(k+1):]], dim=0)
#     train_dataset = Data.TensorDataset(train_X, train_y)
#     train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
#     test_dataset = Data.TensorDataset(test_X, test_y)
#     test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=False)
#     return train_iter, test_iter

def load_rand_iter(samples_X, samples_y, batch_size, filename):
    test_sample = np.loadtxt('../data/DownStreamdata/' + filename + '_warm_node.txt', dtype=int)
    train_X, train_y = [], []
    test_X, test_y = [], []
    for i in range(len(samples_X)):
        if i in test_sample:
            test_X.append(samples_X[i, :].tolist())
            test_y.append((samples_y[i]).tolist())
        else :
            train_X.append(samples_X[i, :].tolist())
            train_y.append((samples_y[i]).tolist())
    train_X = torch.tensor(train_X, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float)
    test_y = torch.tensor(test_y, dtype=torch.long)
    train_dataset = Data.TensorDataset(train_X, train_y)
    train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(test_X, test_y)
    test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=False)
    return train_iter, test_iter


def load_cold_data(input_data, feature, batch_size, filename):
    train_X, train_y = [], []
    test_X, test_y = [], []

    cold_nodes = np.loadtxt('../data/DownStreamdata/' + filename + '_cold_node.txt', dtype=int)
    for line in input_data:
        line = line.split(' ')
        is_train = True
        y = int(line[-1])
        x = []
        for node in line[:-1]:
            if int(node) in cold_nodes:
                is_train = False
            x.extend(feature[int(node)])
        if is_train:
            train_X.append(x)
            train_y.append(y)
        else:
            test_X.append(x)
            test_y.append(y)
    train_X = torch.tensor(train_X, dtype=torch.float)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float)
    test_y = torch.tensor(test_y, dtype=torch.long)
    train_dataset = Data.TensorDataset(train_X, train_y)
    train_iter = Data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(test_X, test_y)
    test_iter = Data.DataLoader(test_dataset, batch_size, shuffle=False)
    return train_iter, test_iter


