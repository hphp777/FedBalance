import argparse
import os, pdb, sys, glob, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models 

# import custom dataset classes
from datasets import XRaysTrainDataset
from datasets import XRaysTestDataset

# import neccesary libraries for defining the optimizers
import config

from Base import client, server
# from MOONBase import client, server

import warnings
import random

warnings.filterwarnings(action='ignore')



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = ResNet50.resnet56() ####
# model = ResNet50_fedalign.resnet56()
model = models.efficientnet_b0(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)
model.to(device)

c_num = 5
com_round = 50
client_weighting = 'Imbalance'

data_dir = "C:/Users/hb/Desktop/data/archive"

central_data = XRaysTrainDataset(data_dir, transform = config.transform, indices=list(range(86336)))
# data0,data1,data2,data3,data4 = torch.utils.data.random_split(central_data, ratios)

length = len(central_data)
ratios = np.round(np.random.dirichlet(np.repeat(1, c_num))*length).astype(int)
indices = list(range(length))
random.shuffle(indices)

### Client num should be greater than 
if sum(ratios) > length:
    ratios[4] -= (sum(ratios) - length)
else:
    ratios[4] += (length - sum(ratios))

indices0 = []
indices1 = []
indices2 = []
indices3 = []
indices4 = []

for i in range(0,ratios[0]):
    indices0.append(indices[i])
for i in range(ratios[0],ratios[0] + ratios[1]):
    indices1.append(indices[i])
for i in range(ratios[0] + ratios[1],ratios[0] + ratios[1] + ratios[2]):
    indices2.append(indices[i])
for i in range(ratios[0] + ratios[1] + ratios[2],ratios[0] + ratios[1] + ratios[2] + ratios[3]):
    indices3.append(indices[i])
for i in range(ratios[0] + ratios[1] + ratios[2] + ratios[3],length):
    indices4.append(indices[i])

XRayTrain_dataset0 = XRaysTrainDataset(data_dir, transform = config.transform, indices=indices0)
XRayTrain_dataset1 = XRaysTrainDataset(data_dir, transform = config.transform, indices=indices1)
XRayTrain_dataset2 = XRaysTrainDataset(data_dir, transform = config.transform, indices=indices2)
XRayTrain_dataset3 = XRaysTrainDataset(data_dir, transform = config.transform, indices=indices3)
XRayTrain_dataset4 = XRaysTrainDataset(data_dir, transform = config.transform, indices=indices4)

central_server = server()
client0 = client(0, XRayTrain_dataset0)
client1 = client(1, XRayTrain_dataset1)
client2 = client(2, XRayTrain_dataset2)
client3 = client(3, XRayTrain_dataset3)
client4 = client(4, XRayTrain_dataset4)

imbalance0 = client0.imbalance
imbalance1 = client1.imbalance
imbalance2 = client2.imbalance
imbalance3 = client3.imbalance
imbalance4 = client4.imbalance

clients = [client0,client1,client2,client3,client4]
imbalances = np.array([imbalance0,imbalance1,imbalance2,imbalance3,imbalance4])
weights = [0] * 5
weight = model.state_dict()
server_auc = []
server_acc = []
best_acc = 0
best_auc = 0

total_data_num = length

cw = []

def set_weight_client(client_weighting):

    if client_weighting == 'DataAmount':
        for i in range(c_num):
            cw.append(len(clients[i].dataset) / total_data_num)
    elif client_weighting == 'Imbalance':
        for i in range(c_num):
            cw.append(imbalances[i] / imbalances.sum())

def draw_auc():

    plt.plot(range(len(server_auc)), server_auc)
    plt.savefig('./results/FedAvg_auc.png')
    plt.clf()

def FL():

    set_weight_client(client_weighting)
    print(cw)

    print("\nCommunication Round 1")

    for i in range(c_num):
        weights[i] = clients[i].train()

    for key in weights[0]:
        weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)])

    # Test
    auc ,acc= central_server.test(weight)
    best_acc = acc
    best_auc =auc
    server_auc.append(auc)
    server_acc.append(acc)

    for r in range(2, com_round+1):

        print("\nCommunication Round " + str(r))

        for i in range(c_num):
            weights[i] = clients[i].train(updated=True, weight=weight)

        for key in weights[0]:
            weight[key] = sum([weights[i][key] * cw[i] for i in range(c_num)]) 

        torch.save(weight, 'C:/Users/hb/Desktop/code/2.FedBalance/Weight/Finals/fedavg_weighted_loss_Imbalance_weight(alpha=1).pth' )

        # Test
        auc, acc = central_server.test(weight)
        if auc > best_auc:
            best_auc = auc
            
        if acc > best_acc:
            best_acc = acc
        server_auc.append(auc)
        server_acc.append(acc)

        print("Best AUC: ", best_auc)
        print("Best Acc: ", best_acc)


    print("AUCs : ", server_auc)
    print("Best AUC: ", best_auc)
    print("Best Acc: ", best_acc)

    return server_auc,server_acc

def CZ():
    
    best_acc = 0
    best_auc = 0

    for i in range(20):

        print("Epoch{}".format(i))
        weight = clients[0].train()
        torch.save(weight, 'C:/Users/hb/Desktop/code/2.TF_to_Torch/Weight/CZ/server_' + str(i) + '.pth' )
        auc, acc = central_server.test(weight)

        if auc > best_auc:
            best_auc = auc
        if acc > best_acc:
            best_acc = acc
    
        server_auc.append(auc)
        server_acc.append(acc)
    
    print("AUCs : ", server_auc)
    print("Best AUC: ", best_auc)
    print("Best Acc: ", best_acc)

if __name__ == '__main__':
    FL()
    draw_auc()