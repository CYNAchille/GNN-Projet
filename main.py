import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
import scipy.ndimage
import torch.nn as nn
import torchvision
from skimage import transform
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
from torch_geometric.data import Data
import data_loading
import preprocess
import model


def train(model, dataset):
    model.train()
    loss = 0
    for data in dataset:
        x = data.x.cuda()
        edge = data.edge_index

        z = model.encode(x, edge)
        loss += model.recon_loss(z, edge)
        # if args.variational:
        #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss / len(dataset))


def test(model, dataset):
    model.eval()
    loss = 0
    for data in dataset:
        x = data.x.cuda()
        edge = data.edge_index
        with torch.no_grad():
            z = model.encode(x, edge)
            loss += model.recon_loss(z, edge)
    return float(loss / len(dataset))

def Adj2index(A):
    nodeA = []
    nodeB = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                nodeA.append(i)
                nodeB.append(j)
    edge_index = torch.tensor([nodeA, nodeB])
    return edge_index


knn_graph = 32
alexnet = torchvision.models.alexnet(pretrained=False)
alexnet.load_state_dict(torch.load('./alexnet-owt-7be5be79.pth'))
imgset = data_loading.load_imgs()
dataset = []

for img in imgset:
    superpixels = slic(img, n_segments=1000)
    avg_values, coord, masks = preprocess.superpixel_Alexnet_features(img, superpixels, alexnet)
    A_spatial = preprocess.spatial_graph(coord, img.shape[:2], knn_graph=knn_graph)
    data = Data()
    data.x = torch.tensor(avg_values, dtype=torch.float32)
    data.edge_index = Adj2index(A_spatial)
    dataset.append(data)

# parameters
out_channels = 2
num_features = dataset[0].x.shape[1]
epochs = 100

# model
my_model = GAE(model.GCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
my_model = my_model.to(device)


# inizialize the optimizer
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.01)


for epoch in range(1, epochs + 1):
    loss_train = train(my_model,dataset)
    print('Epoch: {:03d}, train loss: {:.4f}'.format(epoch, loss_train))