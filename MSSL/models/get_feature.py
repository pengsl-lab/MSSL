import torch
import argparse
from models_class import *
from models_reg import *
from models_sim import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./', help='path of the checkpoint.')
parser.add_argument('--length', type=int, default=2, help='the number of task.')

args = parser.parse_args()
features, adj = load_graph()
features = features.cuda()
adj = adj.cuda()
print("load graph successfully!")
model = torch.load(args.model+'/share.pt')
model = [att.cuda() for att in model]
embeddings = torch.cat([att(features, adj) for att in model], dim=1)
for i in range(args.length):
    model = torch.load(args.model+'/'+str(i)+'.pt')
    model = model.cuda()
    embedding = torch.cat([att(features, adj) for att in model.private], dim=1)
    embeddings = torch.cat([embeddings, embedding], dim=1)
embeddings = embeddings.cpu()
torch.save(embeddings, "feature_"+args.model+'.pt')
print("get feature "+args.model+" successfully!")