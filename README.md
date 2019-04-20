# satellite-building-detection.pytorch

**THIS REPO IS DEPRECATED. PLEASE REFER TO https://github.com/YuansongFeng/satellite_building_detection.pytorch FOR THE LATEST UPDATE.**

This project is on detecting and classifying buildings through satellite images. Satellite images are inputted in raw form with 3 channels, without building boundary information. The model then runs a [Single Shot Detector](https://arxiv.org/abs/1512.02325) to simultaneously propose potential buildings and classify the proposed region into target classes. For now we only classify buildings into two types: residential and non-residential. 

As another objective of the project, we aim to demonstrate the effectiveness of transfer learning across geometric domains. Specifically, we first train the network on building images from all over the world. This generic dataset comes from [Functional Map of the World(fMoW)](https://arxiv.org/abs/1711.07846). We then fine tune our model based on region-specific building images. In our case, we use Wake County satellite images to prompt the model to learn more detailed features. We test our model performance on the satellite images from the specific region that we fine tune the model upon. 

## Requirements
- python 3.6 
- [PyTorch 1.0.0](https://pytorch.org/), for cuda 9.0
- opencv, matplotlib and other requirements are listed in environment.yml

```
$ conda create -n env_name -f environment.yml
```
This creates a new conda environment called env_name and installs all required packages in environment.yml


You need to download pretrained resnet model for both training and evaluation. The models can be downloaded from [here](https://drive.google.com/open?id=0B7fNdx_jAqhtbVYzOURMdDNHSGM), and should be placed in `data/imagenet_weights`.

## Pretrained models 


## Getting started 
First modify the `params.py` file and point `dataset_fmow` and `dataset_wc` to the downloaded dataset. 

The ipython notebook `main.ipynb` provides an overview of the whole project workflow, including data preprocessing, model specification, training, evaluation and visualization. 

If you are running a remote server, ipython notebook allows for remote access in the following fashion.
```
remote_user@remote_server$ ipython notebook --no-browser --port=8999
```
This opens notebook on the remote server on port 8999 without opening a browser. We will instead forward the content to our local server. 

```
local_user@local_server$ ssh -N -f -L localhost:8888:localhost:8999 remote_user@remote_host
```
This creates an ssh shell to our remote server and enable port forwarding from remote server(-N). It also makes the shell go to background(-f). 
After this, open a browser on local server and direct towards `localhost:8888`. 


## Train your own network on fMoW

## Train your own network on Wake County 


## Pre-processed dataset
A truncated version of Functional Map of the World can be accessed [here](https://drive.google.com/open?id=1sdcxiBlFWmbixkP-gSfXJUlSDNWXcHlR). Notice several building types are removed from original dataset as specified in [params.py](https://github.com/YuansongFeng/satellite-building-detection.pytorch/blob/master/params.py#L28-L92). 

Pre processed image data for wake county satellite images can be accessed [here](https://drive.google.com/file/d/1pEBzcPyIl1O2My8IO-E1nndHb7dTFZXD/view?usp=sharing) but notice these images are not open to public, so please open an issue for access. 

To understand the structure of each dataset, please check `dataset.py`. 

## Pretrained models
Pretrained models are provided in `/models`. 


## Referrence
[Functional Map of the World](https://arxiv.org/abs/1711.07846)
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
[Class Activation Map](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
[YOLO](https://arxiv.org/abs/1506.02640)
[Single Shot Detector](https://arxiv.org/abs/1512.02325)

## Resources
[Detailed hands-on experience with Yolo](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
