import torch

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



