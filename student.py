#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

"""

See answers in hw2.pdf

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################

# Following are the only transforms seen to make a positive contribution to accuracy

def transform(mode):

    if mode == 'train':

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(),
        ])

    elif mode == 'test':
        return transforms.ToTensor()

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################

# Network Architecture is a modified Alexnet, see hw2.pdf for details
# Layers that are traditionally part of Alexnet but removed for performance are surrounded by triple apostrophe ''' and denoted (Abandoned)

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional Layer: Convolution -> Batch Normalization -> ReLu Activation -> Max Pooling
        # Fully Connected Layer: Dropout Layer -> Linear Layer -> Batch Normalization -> ReLu Activation
        # Final Fully Connected Layer: Dropout Layer -> Linear Layer

        # Convolutional Layer 1

        self.conv1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2)
        self.normal3 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Convolutional Layer 2

        self.conv2 = nn.Conv2d(32, 96, kernel_size=5, padding=2)
        self.normal4 = nn.BatchNorm2d(96)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Convolutional Layer 3 (Abandoned)

        '''

        self.conv3 = nn.Conv2d(72, 144, kernel_size=3, padding=1)
        self.normal5 = nn.BatchNorm2d(144)
        self.relu3 = nn.ReLU(inplace=True)

        '''

        # Convolutional Layer 3 (Abandoned)

        '''

        self.conv4 = nn.Conv2d(144, 96, kernel_size=3, padding=1)
        self.normal6 = nn.BatchNorm2d(96)
        self.relu4 = nn.ReLU(inplace=True)

        '''
        
        # Convolutional Layer 3 (Abandoned)

        '''

        self.conv5 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.normal7 = nn.BatchNorm2d(96)
        self.relu5 = nn.ReLU(inplace=True)

        '''
        
        # Maxpool Layer 3 (Abandoned)

        '''

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        '''

        # Average Pool Layer

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully-Connected Layer 1

        self.dropout1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(3456, 1728)
        self.normal1 = nn.BatchNorm1d(1728)
        self.relu6 = nn.ReLU(inplace=True)

        # Fully-Connected Layer 2

        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1728, 1728)

        # Fully-Connected Layer 3 (Abandoned)

        '''

        self.normal2 = nn.BatchNorm1d(1728) 
        self.relu7 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(1728, 8)

        '''

    def forward(self, input):

        # Convolutional Layer: Convolution -> Batch Normalization -> ReLu Activation -> Max Pooling
        # Fully Connected Layer: Dropout Layer -> Linear Layer -> Batch Normalization -> ReLu Activation
        # Final Fully Connected Layer: Dropout Layer -> Linear Layer

        # Convolutional Layer 1

        input = self.conv1(input)
        input = self.normal3(input)
        input = self.relu1(input)
        input = self.maxpool1(input)

        # Convolutional Layer 2

        input = self.conv2(input)
        input = self.normal4(input)
        input = self.relu2(input)
        input = self.maxpool2(input)

        # Convolutional Layer 3 (Abandoned)

        '''

        input = self.conv3(input)
        input = self.normal5(input)
        input = self.relu3(input)

        '''

        # Convolutional Layer 4 (Abandoned)

        '''

        input = self.conv4(input)
        input = self.normal6(input)
        input = self.relu4(input)

        '''

        # Convolutional Layer 5 (Abandoned)

        '''

        input = self.conv5(input)
        input = self.normal7(input)
        input = self.relu5(input)

        '''

        # Maxpool Layer 3 (Abandoned)

        '''

        input = self.maxpool3(input)

        '''

        # Average Pool Layer and Flatten for Fully-Connected Layers

        input = self.avgpool(input)
        input = torch.flatten(input, 1)

        # Fully-Connected Layer 1

        input = self.dropout1(input)
        input = self.linear1(input)
        input = self.normal1(input)
        input = self.relu6(input)

        # Fully-Connected Layer 2

        input = self.dropout2(input)
        input = self.linear2(input)

        # Fully-Connected Layer 3 (Abandoned)

        '''

        input = self.normal2(input)
        input = self.relu7(input)
        input = self.linear3(input)

        '''

        return input

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################

# Stochastic Gradient Descent Optimizer chosen as Adam worked poorly
# Cross Entropy Loss Function chosen as well suited to multi-class classification

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0001)

loss_func = F.cross_entropy

############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Xavier Normal Distribution Weight Initialization chosen as it worked well

def weights_init(m):

    # Convolutional Layer Weight Initialization

    net.conv1.weight = torch.nn.init.xavier_normal_(net.conv1.weight)
    net.conv2.weight = torch.nn.init.xavier_normal_(net.conv2.weight)

    # Convolutional Layer Weight Initialization (Abandoned)

    '''

    #net.conv3.weight = torch.nn.init.xavier_normal_(self.conv3.weight)
    #net.conv4.weight = torch.nn.init.xavier_normal_(self.conv4.weight)
    #net.conv5.weight = torch.nn.init.xavier_normal_(self.conv5.weight)

    '''

    # Fully-Connected Layer Weight Initialization

    net.linear1.weight = torch.nn.init.xavier_normal_(net.linear1.weight)
    net.linear2.weight = torch.nn.init.xavier_normal_(net.linear2.weight)

    # Fully-Connected Layer Weight Initialization (Abandoned)

    '''

    net.linear3.weight = torch.nn.init.xavier_normal_(self.linear3.weight)

    '''

    return

scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

############################################################################
#######              Metaparameters and training options              ######
############################################################################

# Kept original train_val_split as it was most efficient
# Chose a small batch size for efficiency and accuracy
# Test accuracy peaked at 300 epochs for current model

dataset = "./data"
train_val_split = 0.9
batch_size = 16
epochs = 300