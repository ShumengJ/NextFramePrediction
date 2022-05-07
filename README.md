# NextFramePrediction
Course Project for ECE-6123 2022Spring \
Team 15: Shumeng Jia, Yuhan Shang, Yuanzi Liu

## About Project
Convolutional neural networks (CNNs) have been widely used in optical flow estimation problems to solve computer vision tasks. Given two frames, the CNNs can be trained to generate the motion directions of each pixel in terms of the optical flow, which can be utilized in predicting the next frame in the video. In this report, we present a hybrid CNN-based structure for both optical flow estimation and next frame prediction, which was firstly proposed by Sedaghat et al [1] in 2017. This multi-tasking structure intelligently trains the networks on different tasks: optical flow estimation and next frame prediction. The output results are affected by the learning outcomes of both sides. We train the networks with real-world video frames and optical flows from the KITTI benchmark [2], hoping to improve their performance in solving real-life problems. With the availability of ground-truth datasets, the networks are trained with supervised learning.


## About Dataset
[KITTI Dataset](http://www.cvlibs.net/datasets/kitti/index.php), is an available real-world dataset with ground truth optical flow as we expect. it is generated by simultaneously recording some real-world scenes with cameras and 3D laser scanners. We use the both 2012 and 2015 version and split it into 400 training samples and 400 test samples. For each training pair, we have two adjacent frames (note as t_{0} and t_{1}) as well as the corresponding optical flow and the frame immediately after them (note as t_{2}) to train and optimize the model. For test pairs, we only have the ground truth for the next frame prediction, but no optical flow estimation. 10 percent of training data is used for validation purposes. 

## About the model
A Hybrid learning model implement based on [this work](https://arxiv.org/abs/1612.03777)
\
<img src="https://www.researchgate.net/profile/Nima-Sedaghat/publication/316714926/figure/fig2/AS:491193416065024@1494121044234/We-improve-CNN-based-optical-flow-estimation-in-real-videos-by-adding-the-extra.png" width="800"  />

## File Descriptions
### CommentsResults.ipynb
Background info, comments, some results and examples can be found in this notebook.
### dataset.py
To generate image pairs from raw multiview dataset, run this file.
### model.py
Construct the hybrid learning model.
### train.py
To train the model, run this file.
### predict.py
To see how the model performs and generate next frame, run this file.
### loss.py
Loss functions used in our project
### helper_func.py
Functions that help training and validating.
### NFmodelEpoch99.pth
Load the parameters and settings in this file to use our best model.

## Performance
<img src="https://raw.githubusercontent.com/ShumengJ/NextFramePrediction/main/result_a3.png" width="800"  />

## Reference
[1] N. Sedaghat, ‘‘Hybrid Learning of Optical Flow and Next Frame Prediction to Boost Optical Flow in the Wild,’’ 2016, arXiv:1612.03777. Accessed: Apr. 7, 2017. [Online]. Available: https://arxiv.org/abs/1612. 03777 \
[2] KITTI, [online].Available:http://www.cvlibs.net/datasets/kitti
