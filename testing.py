import torch.nn as nn
import torch.nn.functional as F
import pandas as pd                     # to process our data
import matplotlib.pyplot as plt         # graphing
import numpy as np                      # matrices

import torch
import torchvision                      # for MNIST dataset/working with images

import os
from PIL import Image
import cv2

class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()
        self.act = nn.PReLU()
        self.act2 = nn.Sigmoid()
        
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride = 2,  padding=2),
            nn.BatchNorm2d(32),
            self.act,
            nn.MaxPool2d(2)
        )
    
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride = 2,  padding=2),
            nn.BatchNorm2d(64),
            self.act,
            nn.MaxPool2d(2)
        )

        self.convBlock3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride = 2,  padding=2),
            nn.BatchNorm2d(128),
            self.act,
            nn.MaxPool2d(2)
        )
        self.convBlock4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, stride = 2,  padding=2),
            nn.BatchNorm2d(256),
            self.act,
            nn.MaxPool2d(2)
        )

        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.convBlock4(x)
        x = x.view(-1, 256 * 1 * 1)
        x = self.drop(self.act2(self.fc1(x)))
        x = self.drop(self.act2(self.fc2(x)))
        x = self.fc3(x)
        return x.squeeze()

def predict(path):
    cnn_model = torch.load("model.pth") 
    mean = 103.303632997174
    std = 95.32723806245423
    image = cv2.imread(path)
    image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)[None]
    return (cnn_model(image.cuda()).item() * std) + mean