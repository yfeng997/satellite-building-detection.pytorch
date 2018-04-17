import torch
import cv2
import numpy as np

def getCAM(X, y, model):
    """
    Compute a class activation map on image X for class y.

    Input:
    - X: Input images; Tensor of shape (N, 1, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the activation map.

    Returns:
    - class activatino map: A Tensor of shape (N, H, W, 3) giving the class activation maps for the input
    images.
    """
    if len(list(X.size())) == 0:
        return []
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # get weight for feature maps from last fc layer
    params = list(model.parameters())
    weights = np.squeeze(params[-2].data.cpu().numpy())
    
    # get feature maps (N, C, H, W)
    feature_maps = []
    def hook_feature_map(module, input, output):
        feature_maps.append(output.data.cpu().numpy())
    # store feature maps during forward pass
    model._modules.get('features').register_forward_hook(hook_feature_map)
    
    if torch.cuda.is_available():
        X_var = torch.autograd.Variable(X.cuda(async=True))
    else:
        X_var = torch.autograd.Variable(X)
    # forward pass input variable and obtain features
    output = model(X_var)
    
    # combine feature maps with corresponding weights
    cams = []
    N, C, H, W = X.size()
    for i in range(N):
        maps = feature_maps[0][i]
        weight = weights[y[i]]
        c, h, w = maps.shape
        cam = weight.dot(maps.reshape((c, h*w)))
        cam = cam.reshape(h, w)
        cam = np.uint8((cam-np.min(cam))/np.max(cam) * 255)
        cams.append(cv2.resize(cam, (W, H)))
    
    return cams

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 1, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    finalgrad = torch.FloatTensor([1.0 for i in range(X.size()[0])])
    # Wrap the input tensors in Variables
    if torch.cuda.is_available():
        X_var = torch.autograd.Variable(X.cuda(async=True),requires_grad=True)
        y_var = torch.autograd.Variable(y.cuda(async=True))
        finalgrad = finalgrad.cuda(async=True)
    else:
        X_var = torch.autograd.Variable(X, requires_grad=True)
        y_var = torch.autograd.Variable(y)
    
    # Forward Pass
    output = model(X_var)
    
    # Collect the scores to correct class
    output = output.gather(1, y_var.view(-1,1)).squeeze()
    
    # Backward pass
    finalgrad = torch.FloatTensor([1.0 for i in range(X.size()[0])]).cuda(async=True)
    output.backward(finalgrad)
    
    saliency = X_var.grad.data.abs()
    return saliency



