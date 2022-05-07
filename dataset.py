 
# -*- coding: utf-8 -*-
# @Time : 2021/5/8
# @Author : Shumeng Jia
# @File : dataset.py
# @function : Data preparation

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



class KITTIDataset(Dataset):
    def __init__(self, img0_path_list, img1_path_list, flow_label_path_list, frame_label_path_list):
        self.img0_path_list = img0_path_list
        self.img1_path_list = img1_path_list
        self.flow_label_path_list = flow_label_path_list
        self.frame_label_path_list = frame_label_path_list

        
        self.img_list, self.flow_list, self.frame_list = self.preprocess()
        
    def __len__(self):
        return len(self.img_list)

    def preprocess(self):

        img_list, flow_list, frame_list = [], [], []
        for idx in tqdm(range(len(self.flow_label_path_list))):
            img0 = cv2.imread(self.img0_path_list[idx])
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.imread(self.img1_path_list[idx])
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img = np.concatenate((img0,img1), axis=2)
            img = img[40:360,0:1024,:]
            flow = np.float64(np.array(cv2.imread(self.flow_label_path_list[idx], cv2.IMREAD_UNCHANGED)))
            flow = (np.concatenate(([flow[:,:,2]],[flow[:,:,1]]), axis=0)  - 2**15) / 64
            flow = flow[:,40:360,0:1024]
            frame = cv2.imread(self.frame_label_path_list[idx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frame = frame[40:360,0:1024,:]        
            
            img_list.append(img)
            flow_list.append(flow)
            frame_list.append(frame)
        return img_list, flow_list, frame_list
                    
                    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        flow = self.flow_list[idx]
        frame = self.frame_list[idx]

        img = torch.Tensor(img).permute(2,0,1)
        flow = torch.Tensor(flow)
        frame = torch.Tensor(frame).permute(2,0,1)      
        
        
        return img/255., flow/255., frame/255