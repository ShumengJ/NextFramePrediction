 
# -*- coding: utf-8 -*-
# @Time : 2021/5/8
# @Author : Shumeng Jia
# @File : train.py
# @function : training and evaluation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn import init
from torchvision import transforms
from torch.utils.data import Dataset

from model import NextFlow
from loss import EPEMSELoss
from dataset import KITTIDataset



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = NextFlow(input_channels=6, batchNorm=True)
net.to(device)  

# Shows the number of parameters in the network
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Number of parameters in network: ', n_params)


######### data preprocessing #########

# image pairs that have flow ground truth

img0_list = sorted(glob.glob("./data/Training/a/input/*_10.png"))
img1_list = sorted(glob.glob("./data/Training/a/input/*_11.png"))
flow_label_list = sorted(glob.glob("./data/Training/a/output-flow/*.png"))
frame_label_list = sorted(glob.glob("./data/Training/a/output-12/*.png"))
assert len(img0_list) == len(flow_label_list)
print ("Collected {} images".format(len(img1_list)))


# image pairs that without floe ground truth
# only used to update the second branch (frame branch)

img0_listb = sorted(glob.glob("./data/Training/b/input/*_10.png"))
img1_listb = sorted(glob.glob("./data/Training/b/input/*_11.png"))
frame_label_listb = sorted(glob.glob("./data/Training/b/output-12/*.png"))
assert len(img0_listb) == len(frame_label_listb)
print ("Collected {} images".format(len(img1_listb)))
total_img = len(img1_list)
total_list = np.arange(0,total_img).tolist()



optimizer = torch.optim.Adam(net.parameters(),lr=LR)
criterion = EPEMSELoss()

# Specify number of epochs, image scale factor, batch size and learning rate
NUM_EPOCH = 50 
BATCH_SIZE = 8 
LR = 0.0001
SAVE_PATH = "./model/"

# Lists used for plotting loss
train_loss_list = []
val_loss_list = []



######### Start training #########
for epoch in range(NUM_EPOCH):    
    
    #cross validation
    val_list = random.sample(total_list, 32)
    train_list = list(set(total_list)-set(val_list))
    
    train_dataset = KITTIDataset(
        np.array(img0_list)[train_list].tolist(), 
        np.array(img1_list)[train_list].tolist(), 
        np.array(flow_label_list)[train_list].tolist(), 
        np.array(frame_label_list)[train_list].tolist())
    
    val_dataset = KITTIDataset(
        np.array(img0_list)[val_list].tolist(), 
        np.array(img1_list)[val_list].tolist(), 
        np.array(flow_label_list)[val_list].tolist(), 
        np.array(frame_label_list)[val_list].tolist())
    
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True, 
                                           num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=BATCH_SIZE, 
                                             shuffle=False, 
                                             num_workers=0)

    
    
    loss = train_epoch(net, train_loader, optimizer, criterion, epoch)
    epe, val_loss = eval_epoch(net, val_loader, EPE, criterion, epoch)
    
    # Record losses for each epoch
    train_loss_list.append(loss.item())
    val_loss_list.append(val_loss.item())
    
    # Save the model every 10 epoch
    if os.path.isdir(SAVE_PATH) and epoch%10 == 0:
        torch.save(net.state_dict(),SAVE_PATH + NFmodelEpoch{}.pth'.format(epoch))
    # else:
        # os.makedirs(model_save_path, exist_ok=True)
        # torch.save(net.state_dict(),SAVE_PATH + 'PedSegEpoch{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved to {}'.format(epoch, SAVE_PATH + NFmodelEpoch{}.pth'.format(epoch))) 




######### Plot training loss and validation loss #########

plt.figure(figsize=(19,9))
plt.plot(train_loss_list,'.-', label='train loss')
plt.plot(val_loss_list,'.-', label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('History')
plt.legend()