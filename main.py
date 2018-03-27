import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset

import numpy as np
import cv2
#import matplotlib.pyplot as plt

from cs231n.data_utils import *
import params

# Load parameters
params = vars(params)
class_names = params['class_names']
dtype = params['dtype']

torch.set_num_threads(params['num_threads'])

data = load_mini_fmow(params, batch_size=50)

# Unpack training and validation dat
X_train = torch.from_numpy(data['X_train'])
y_train = torch.from_numpy(data['y_train'])
train_data = TensorDataset(X_train, y_train)
loader_train = DataLoader(train_data, batch_size=64)

# X_val = torch.from_numpy(data['X_val'])
# y_val = torch.from_numpy(data['y_val'])
# val_data = data_utils.TensorDataset(X_val, y_val)
# loader_val = DataLoader(val_data, batch_size=64)

# # Unpack test data
# X_test = torch.from_numpy(data['X_test'])
# y_test = torch.from_numpy(data['y_test'])
# test_data = data_utils.TensorDataset(X_test, y_test)
# loader_test = DataLoader(test_data, batch_size=64)

# Print out all the keys and values from the data dictionary
for k, v in data.items():
    print(k, type(v), v.shape, v.dtype)


# CNN Model (3 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=6, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(12*12*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
#     200*200  
#     98*98
#     49*49
#     24*24
#     12*12
#     11111---1

# Params
batch_size = params['batch_size']
num_epochs = params['num_epochs']
learning_rate = params['learning_rate']
print_every = 100

model = CNN().type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# Define Train function
def train(model, loss_fn, optimizer, loader, epochs=1):
    for epoch in range(epochs):
        # Set on Training mode
        model.train()
        for t, (x, y) in enumerate(loader):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype))
            socres = model(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

# Train the model
train(model, loss_fn, optimizer, loader_train, epochs=num_epochs)
# Check accuracy of model
check_accuracy(model, loader_val)

