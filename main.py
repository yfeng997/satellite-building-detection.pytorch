import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset

import numpy as np
import cv2 
import os
from memory_profiler import profile

from cs231n.data_utils import *
import params

torch.set_num_threads(1)
#image = cv2.imread("/home/feng/data/fMoW-rgb/train/airport/airport_0/airport_0_2_rgb.jpg", 0)
#print(image.shape)
#img = cv2.resize(image,(200,200)).astype(np.float32)
#print(img.shape)
#print(type(img))


#class_names = ['false_detection','residential','non_residential']
#dtype = torch.FloatTensor


data = load_mini_fmow('/home/feng/data/fMoW-rgb', params, dtype=np.float32, batch_size=100)

#for k, v in data.items():
#    print(k, type(v), v.shape, v.dtype)

