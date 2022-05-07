 
# -*- coding: utf-8 -*-
# @Time : 2021/5/8
# @Author : Shumeng Jia
# @File : predict.py
# @function : Function used to predict and evaluate on test data


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


def predict_img(net, img):
    # set the mode of your network to evaluation
    net.eval()
    img = img.to(device)

    with torch.no_grad():
       
        img = img.unsqueeze(0)
        pred, pred0 = net(img)
        pred = pred * 255
        pred0 = pred0 * 255

    return pred, pred0



def display_results(net, input_imgs):
    
      plt.figure(figsize = (16,9))
    
      pred, pred0 = predict_img(net=net, img=input_imgs[0])

      plt.subplot(3,2,1)
      pred = flow_to_image(np.array(pred.squeeze(0).cpu()))
      plt.imshow(pred.astype(np.uint8))
      plt.title('pred flow')
      plt.subplot(3,2,2)
      acflow = flow_to_image(np.array(input_imgs[1]))
      plt.imshow(acflow)
      plt.title('actual flow')

      plt.subplot(3,2,3)
      time0 = np.array(input_imgs[0][0:3].permute(1,2,0).cpu())*255
      plt.imshow(time0.astype(np.uint8))
      plt.title('time 0 frame')
      plt.subplot(3,2,4)
      time1 = np.array(input_imgs[0][3:6].permute(1,2,0).cpu())*255
      plt.imshow(time1.astype(np.uint8))
      plt.title('time 1 frame')

      plt.subplot(3,2,5)
      plt.imshow(np.array(pred0.squeeze(0).permute(1,2,0).cpu()).astype(np.uint8))
      plt.title('pred next frame')
      plt.subplot(3,2,6)
      plt.imshow(np.array(input_imgs[2].permute(1,2,0).cpu()))
      plt.title('actual next frame')