import argparse
import os, pdb, sys, glob, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torchvision.models as models 
import importlib
importlib.reload(models)

# import custom dataset classes
from datasets import XRaysTrainDataset, GANLoader
from datasets import XRaysTestDataset

# import neccesary libraries for defining the optimizers
import torch.optim as optim

from FedAvg import fit
# from FedAlign import fit
import matplotlib.pyplot as plt
import config

import warnings

warnings.filterwarnings(action='ignore')

def q(text = ''): # easy way to exiting the script. useful while debugging
    print('> ', text)
    sys.exit()

class weighted_loss():

    def __init__(self, pos_weights, neg_weights):
        self.pos_weights = pos_weights
        self.neg_weights = neg_weights

    def __call__(self, y_pred, y_true, epsilon=1e-7):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        sigmoid = nn.Sigmoid()
        
        for i in range(len(self.pos_weights)):
            # for each class, add average weighted loss for that class 
            loss_pos =  -1 * torch.mean(self.pos_weights[i] * y_true[:, i] * torch.log(sigmoid(y_pred[:, i]) + epsilon))
            loss_neg =  -1 * torch.mean(self.neg_weights[i] * (1 - y_true[:, i]) * torch.log(1 -sigmoid( y_pred[:, i]) + epsilon))
            loss += loss_pos + loss_neg
        return loss

class server():

    def __init__(self):

        parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself ! Cool huh ? :D')
        parser.add_argument('--data_path', type = str, default = '.', help = 'This is the path of the training data')
        parser.add_argument('--bs', type = int, default = 32, help = 'batch size')
        parser.add_argument('--lr', type = float, default = 1e-6, help = 'Learning Rate for the optimizer')
        parser.add_argument('--stage', type = int, default = 1, help = 'Stage, it decides which layers of the Neural Net to train')
        parser.add_argument('--loss_func', type = str, default = 'BCE', choices = {'BCE', 'FocalLoss'}, help = 'loss function')
        parser.add_argument('-r','--resume', default= False ,action = 'store_true') # args.resume will return True if -r or --resume is used in the terminal
        parser.add_argument('--ckpt', type = str, help = 'Path of the ckeckpoint that you wnat to load')
        parser.add_argument('-t','--test', action = 'store_true')   # args.test   will return True if -t or --test   is used in the terminal
        self.args = parser.parse_args()

        self.args.test = True
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # select model

        # ResNet
        # self.model = models.resnet50(pretrained=True)
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 15) # 15 output classes 

        # EfficientNetB0
        self.model = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)

        self.model.to(self.device)
        
        data_dir = "C:/Users/hb/Desktop/data/archive"
        self.batch_size = 32
        self.lr = self.args.lr
        self.stage = 1
        train_percentage = 0.8
        XRayTrain_dataset = XRaysTrainDataset(data_dir, transform = config.transform, indices=list(range(10000)))
        self.XRayTest_dataset = XRaysTestDataset(data_dir, transform = config.transform)
        
        train_dataset, val_dataset = torch.utils.data.random_split(XRayTrain_dataset, [int(len(XRayTrain_dataset)*train_percentage), len(XRayTrain_dataset)-int(len(XRayTrain_dataset)*train_percentage)])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size, shuffle = True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size, shuffle = not True)
        self.test_loader = torch.utils.data.DataLoader(self.XRayTest_dataset, batch_size = self.batch_size, shuffle = not True)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.lr)

        ori_ds_cnt = self.XRayTest_dataset.get_ds_cnt()
        total_ds_cnt = np.array(ori_ds_cnt)

        pos_freq = total_ds_cnt / total_ds_cnt.sum()
        neg_freq = 1 - pos_freq

        pos_weights = neg_freq
        neg_weights = pos_freq

        if self.args.loss_func == 'FocalLoss': # by default
            from losses import FocalLoss
            self.loss_fn = FocalLoss(device = self.device, gamma = 2.).to(self.device)
        elif self.args.loss_func == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        self.loss_fn = weighted_loss(pos_weights,neg_weights)
        

    def test(self,weight):

        ckpt = 'C:/Users/hb/Desktop/code/FedAvg/models/C0_stage1_1e-05.pth'

        if ckpt == None:
            q('ERROR: Please select a checkpoint to load the testing model from')
        
        data_dir = "C:/Users/hb/Desktop/data/archive"
        print('\ncheckpoint loaded: {}'.format(ckpt))
        ckpt = torch.load('C:/Users/hb/Desktop/code/2.TF_to_Torch/C0_stage1_1e-05.pth') 
        # self.model = ckpt['model']

        # since we are resuming the training of the model
        epochs_till_now = ckpt['epochs']
        # self.model = ckpt['model']
        self.model.load_state_dict(weight)
        self.model.eval()
    
        # loading previous loss lists to collect future losses
        losses_dict = ckpt['losses_dict'] 

        auc, acc = fit(self.device, self.train_loader, self.val_loader,    
                                        self.test_loader, self.model, self.loss_fn, 
                                        self.optimizer, losses_dict,
                                        epochs_till_now = epochs_till_now, epochs = 3,
                                        log_interval = 25, save_interval = 1,
                                        lr = self.lr, bs = self.batch_size, stage = self.stage,
                                        test_only = self.args.test)
        
        return auc, acc


class client():

    def __init__(self, c_num = None, dataloader = None):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself ! Cool huh ? :D')
        parser.add_argument('--data_path', type = str, default = '.', help = 'This is the path of the training data')
        parser.add_argument('--bs', type = int, default = 32, help = 'batch size')
        parser.add_argument('--lr', type = float, default = 1e-5, help = 'Learning Rate for the optimizer')
        parser.add_argument('--stage', type = int, default = 1, help = 'Stage, it decides which layers of the Neural Net to train')
        parser.add_argument('--loss_func', type = str, default = 'BCE', choices = {'BCE', 'FocalLoss'}, help = 'loss function')
        parser.add_argument('-r','--resume', action = 'store_true') # args.resume will return True if -r or --resume is used in the terminal
        parser.add_argument('--ckpt', type = str, help = 'Path of the ckeckpoint that you wnat to load')
        parser.add_argument('-t','--test', action = 'store_true')   # args.test   will return True if -t or --test   is used in the terminal
        self.args = parser.parse_args()
        disease = ['Cardiomegaly','Emphysema','Effusion','Hernia','Infiltration','Mass','Nodule','Atelectasis','Pneumothorax','Pleural_Thickening','Pneumonia','Fibrosis','Edema','Consolidation']


        # define the learning rate
        self.c_num = c_num
        self.lr = self.args.lr
        self.stage = self.args.stage

        self.local_epoch = 2

        # select model
        # ResNet
        # self.model = models.resnet50(pretrained=True)
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 15) # 15 output classes 

        # EfficientNetB0
        self.model = models.efficientnet_b0(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=14)

        self.model.to(self.device)
        self.batch_size = self.args.bs
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data_dir = "C:/Users/hb/Desktop/data/archive"
        
        # mention the path of the data
        self.XRayTrain_dataset = dataloader
        XRayTest_dataset = XRaysTestDataset(data_dir, transform = config.transform)
        self.dataset = self.XRayTrain_dataset
        # self.dataset = self.XRayTrain_dataset
        train_percentage = 0.8
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*train_percentage), len(self.dataset)-int(len(self.dataset)*train_percentage)])

        ori_ds_cnt = self.XRayTrain_dataset.get_ds_cnt()

        # total_ds_cnt = np.array(ori_ds_cnt) + np.array(gan_ds_cnt)
        total_ds_cnt = np.array(ori_ds_cnt)

        pos_freq = total_ds_cnt / total_ds_cnt.sum()
        neg_freq = 1 - pos_freq

        pos_weights = neg_freq
        neg_weights = pos_freq

        batch_size = self.args.bs # 128 by default
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)
        self.test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size = batch_size, shuffle = not True)

        #print(len(self.XRayTrain_dataset))
        #print(len(self.XRayTrain_dataset)+len(self.GANTrain_dataset))

        print('\n-----Initial Dataset Information({})-----'.format(self.c_num))
        print('num images in train_dataset   : {}'.format(len(train_dataset)))
        print('num images in val_dataset     : {}'.format(len(val_dataset)))
        print('num images in XRayTest_dataset: {}'.format(len(XRayTest_dataset)))
        print('-------------------------------------')

        # define the loss function
        if self.args.loss_func == 'FocalLoss': # by default
            from losses import FocalLoss
            self.loss_fn = FocalLoss(device = self.device, gamma = 2.).to(self.device)
        elif self.args.loss_func == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        # self.loss_fn = weighted_loss(pos_weights, neg_weights)

        # Plot the disease distribution
        plt.figure(figsize=(8,4))
        plt.title('Client{} Disease Distribution'.format(c_num), fontsize=20)
        plt.bar(disease,total_ds_cnt)
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.40)
        plt.xticks(rotation = 90)
        plt.xlabel('Diseases')
        plt.savefig('Client{}_disease_distribution.png'.format(c_num))
        plt.clf()



    def count_parameters(self, model): 
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_parameters/1e6 # in terms of millions

    def q(text = ''): # easy way to exiting the script. useful while debugging
        print('> ', text)
        sys.exit()

    def staging(self, stage = 1, resume = True, ckpt = None, c_num = None):

        # initialize the model if not args.resume
        self.stage = stage
        self.args.resume = resume
        self.args.ckpt = ckpt
        
        # mention the path of the data
        train_percentage = 0.8
        train_dataset, val_dataset = torch.utils.data.random_split(self.XRayTrain_dataset, [int(len(self.XRayTrain_dataset)*train_percentage), len(self.XRayTrain_dataset)-int(len(self.XRayTrain_dataset)*train_percentage)])

        batch_size = self.args.bs # 128 by default
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)

        for name, param in self.model.named_parameters(): # all requires_grad by default, are True initially
            param.requires_grad = True 

        # since we are not resuming the training of the model
        epochs_till_now = 0

        # making empty lists to collect all the losses
        losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}

        if not self.args.test:
            # checking the layers which are going to be trained (irrespective of args.resume)
            trainable_layers = []
            for name, param in self.model.named_parameters():
                if param.requires_grad == True:
                    layer_name = str.split(name, '.')[0]
                    if layer_name not in trainable_layers: 
                        trainable_layers.append(layer_name)
            print('following are the trainable layers...')
            print(trainable_layers)

            # print('\nwe have {} Million trainable parameters here in the {} model'.format(self.count_parameters(self.model), self.model.__class__.__name__))
        
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.lr)

        # print("Client{} Data Number".format(self.c_num),len(self.train_loader))

        weight = fit(self.device, self.train_loader, self.val_loader,    
                                        self.test_loader, self.model, self.loss_fn, 
                                        self.optimizer, losses_dict,
                                        epochs_till_now = epochs_till_now, epochs = self.local_epoch,
                                        log_interval = 25, save_interval = 1,
                                        lr = self.lr, bs = self.batch_size, stage = stage,
                                        test_only = self.args.test, c_num = self.c_num)

        return weight

    def train(self, updated = False, weight = None):
        print("\nClient" + str(self.c_num) + " Staging==============================================")
        if updated == True:
            self.model.load_state_dict(weight)
        weight = self.staging(stage = 1, resume = False, c_num = self.c_num)
        # for i in range(2,5):
        #     weight = self.staging(stage = i, resume = True, ckpt = 'C{}_stage{}_1e-05.pth'.format(self.c_num, i-1))
        return weight