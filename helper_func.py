# -*- coding: utf-8 -*-
# @Time : 2021/5/8
# @Author : Shumeng Jia
# @File : helper_func.py
# @function : Auxiliary training function


import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
from numpy.core.multiarray import concatenate


def train_epoch(net, data_loader, optimizer, criterion, epoch):

    net.to(device)
    net.train()
    
    loss_stat = []
    for i, img_labels in enumerate(data_loader):
        
        img, flow, frame = img_labels
        img = img.to(device)
        flow = flow.to(device)
        frame = frame.to(device)

        output, output0 = net(img)

        # Compute loss and perform update of gradients
        loss = criterion(output, flow)[0] + criterion(output0, frame)[1]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_stat += [loss.item()]*img.shape[0]
        print("\r training...{:d}/{:d} ".format(i+1, len(data_loader)), end='',  flush=True)
    
    print ("\n Epoch {}: Loss: {:.3f}".format(epoch, np.mean(loss_stat))) 
    
    return np.mean(loss_stat)


def eval_epoch(net, data_loader, metric, criterion, epoch):

    net.eval()
    
    epe_stat = []
    val_loss_stat = []
    for i, img_labels in enumerate(data_loader):
        img, flow, frame = img_labels

        img = img.to(device)
        flow = flow.to(device)
        frame = frame.to(device)
        
        with torch.no_grad():
            pred, pred0 = net(img)
            val_loss = criterion(pred, flow)[0] + criterion(pred0, frame)[1]

        epe = EPE(pred, flow) 
        epe_stat += [epe.item()]*img.shape[0]
        val_loss_stat += [val_loss.item()]*img.shape[0]
        
    print ("EPE: {:.3f}  Val Loss: {:.3f} ".format(np.mean(epe_stat), np.mean(val_loss_stat)))
    
    return np.mean(epe_stat),  np.mean(val_loss_stat)