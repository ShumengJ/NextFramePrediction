 
# -*- coding: utf-8 -*-
# @Time : 2021/5/8
# @Author : Shumeng Jia
# @File : loss.py
# @function : Loss Functions used to update the model


import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
import cv2
from numpy.core.multiarray import concatenate


def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
    def forward(self, output, target):
        lossvalue = ((output-target)**2).mean()
        return lossvalue
    
class L1L2Loss(nn.Module):
    def __init__(self):
        super(L1L2Loss, self).__init__()
        self.lossl1 = L1()
        self.lossl2 = L2()
        self.loss_labels = ['L1', 'L2']

    def forward(self, output, target):
        lossl1 = self.lossl1(output, target)
        lossl2 = self.lossl2(output, target)
        return [lossl1, lossl2]

class EPEMSELoss(nn.Module):
    def __init__(self):
        super(EPEMSELoss, self).__init__()
        self.lossl1 = L2()
        self.lossl2 = MSE()
        self.loss_labels = ['EPE', 'MSE']

    def forward(self, output, target):
        lossl1 = self.lossl1(output, target)
        lossl2 = self.lossl2(output, target)
        return [lossl1, lossl2]