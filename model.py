  
# -*- coding: utf-8 -*-
# @Time : 2021/5/8
# @Author : Shumeng Jia
# @File : model.py
# @function : Implementation of NextFlow network structure


import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torchvision import transforms
from torch.utils.data import Dataset
from numpy.core.multiarray import concatenate

%matplotlib inline



def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


def predict_frame(in_planes):
    return nn.Conv2d(in_planes,3,kernel_size=3,stride=1,padding=1,bias=True)



class NextFlow(nn.Module):
    def __init__(self, input_channels = 6, batchNorm=True):
        super(NextFlow,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,  input_channels,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

	###################### Flow ####################

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

	###################### Frame ####################

        self.deconv5_0 = deconv(1024,512)
        self.deconv4_0 = deconv(1027,256)
        self.deconv3_0 = deconv(771,128)
        self.deconv2_0 = deconv(387,64)

        self.predict_flow6_0 = predict_frame(1024)
        self.predict_flow5_0 = predict_frame(1027)
        self.predict_flow4_0 = predict_frame(771)
        self.predict_flow3_0 = predict_frame(387)
        self.predict_flow2_0 = predict_frame(195)

        self.upsampled_flow6_to_5_0 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4_0 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3_0 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2_0 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample1_0 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):

        out_conv1 = self.conv1(x)
        # print(out_conv1.shape)
        out_conv2 = self.conv2(out_conv1)
        # print(out_conv2.shape)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        # print(out_conv3.shape)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # print(out_conv4.shape)
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # print(out_conv5.shape)
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        # print(out_conv6.shape)
        # print('--------------------------')

	###### Branch1 ######

        flow6       = self.predict_flow6(out_conv6)
        # print(flow6.shape)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        # print(flow6_up.shape)
        out_deconv5 = self.deconv5(out_conv6)
        # print(out_deconv5.shape)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        # print(concat5.shape)
        flow5       = self.predict_flow5(concat5)
        # print(flow5.shape)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        # print(flow5_up.shape)
        out_deconv4 = self.deconv4(concat5)
        # print(out_deconv4.shape)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        # print(concat4.shape)
        flow4       = self.predict_flow4(concat4)
        # print(flow4.shape)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        # print(flow4_up.shape)
        out_deconv3 = self.deconv3(concat4)
        # print(out_deconv3.shape)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        # print(concat3.shape)
        flow3       = self.predict_flow3(concat3)
        # print(flow3.shape)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        # print(flow3_up.shape)
        out_deconv2 = self.deconv2(concat3)
        # print(out_deconv2.shape)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        # print(concat2.shape)
        flow2 = self.predict_flow2(concat2)
        # print(flow2.shape)
        flow1 = self.upsample1(flow2)
        # print(flow1.shape)
        # print('---------------------------------------------------')

	##### Branch2 #####

        frame6       = self.predict_flow6_0(out_conv6)
        # print(frame6.shape)
        frame6_up    = self.upsampled_flow6_to_5_0(frame6)
        # print(frame6_up.shape)
        out_deconv5_0 = self.deconv5_0(out_conv6)
        # print(out_deconv5_0.shape)
        
        concat5_0 = torch.cat((out_conv5,out_deconv5_0,frame6_up),1)
        # print(concat5_0.shape)
        frame5       = self.predict_flow5_0(concat5_0)
        # print(frame5.shape)
        frame5_up    = self.upsampled_flow5_to_4_0(frame5)
        # print(frame5_up.shape)
        out_deconv4_0 = self.deconv4_0(concat5_0)
        # print(out_deconv4_0.shape)

        concat4_0 = torch.cat((out_conv4,out_deconv4_0,frame5_up),1)
        # print(concat4_0.shape)
        frame4       = self.predict_flow4_0(concat4_0)
        # print(frame4.shape)
        frame4_up    = self.upsampled_flow4_to_3_0(frame4)
        # print(frame4_up.shape)
        out_deconv3_0 = self.deconv3_0(concat4_0)
        # print(out_deconv3_0.shape)
        
        concat3_0 = torch.cat((out_conv3,out_deconv3_0,frame4_up),1)
        # print(concat3_0.shape)
        frame3       = self.predict_flow3_0(concat3_0)
        # print(frame3.shape)
        frame3_up    = self.upsampled_flow3_to_2_0(frame3)
        # print(frame3_up.shape)
        out_deconv2_0 = self.deconv2_0(concat3_0)
        # print(out_deconv2_0.shape)

        concat2_0 = torch.cat((out_conv2,out_deconv2_0,frame3_up),1)
        # print(concat2_0.shape)
        frame2 = self.predict_flow2_0(concat2_0)
        # print(frame2.shape)
        frame1 = self.upsample1_0(frame2)
        # print(frame1.shape)
        # print('=====================================================')


        if self.training:
            return flow1, frame1  
        else:
            return flow1, frame1