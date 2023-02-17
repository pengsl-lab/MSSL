import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer

class MGTA(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, length, ntask):
        """Dense version of GAT."""
        super(MGTA, self).__init__()
        self.dropout = dropout

        self.private = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.private):
            self.add_module('attention_{}'.format(i), attention)
        
        self.share = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.share):
            self.add_module('attention2_{}'.format(i), attention)
        self.share_classifier = nn.Linear(nhid*nheads, ntask)
        
        self.classifier = nn.Linear(nhid*nheads*length*2, 1)

        
    def forward(self, features, adj, path, task):
        x = F.dropout(features, self.dropout, training=self.training)
        private_x = torch.cat([att(x, adj) for att in self.private], dim=1)
        share_x = torch.cat([att(x, adj) for att in self.share], dim=1)
        private_x = F.dropout(private_x, self.dropout, training=self.training)
        share_x = F.dropout(share_x, self.dropout, training=self.training)
        
        share_feature = torch.cat([share_x[node, :].view(1, -1) for p in path for node in p], dim=0)
        node_task = self.share_classifier(share_feature)
        node_task = F.softmax(torch.sigmoid(node_task), dim=1)
        adv_loss = F.cross_entropy(node_task, task)
        
        private_feature = torch.cat([private_x[node, :].view(1, -1) for p in path for node in p], dim=0)
        diff = share_feature.t().matmul(private_feature)
        diff_loss = (diff**2).sum()
        
        x = torch.cat([share_feature, private_feature], dim=1).view(-1, share_feature.shape[1]*path.shape[1]*2)
        x = F.sigmoid(self.classifier(x))
        
        return x, adv_loss, diff_loss
