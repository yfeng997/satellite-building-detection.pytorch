import fire
import os
import time
import torch
from torchvision import datasets, transforms
from densenet_efficient import DenseNetEfficient
import numpy as np
from meter import AverageMeter, ConfusionMeter


# class AverageMeter(object):
#     """
#     Computes and stores the average and current value
#     Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
#     """
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# class ConfusionMeter(object):
#     """
#     The ConfusionMeter constructs a confusion matrix for a multi-class
#     classification problems. It does not support multi-label, multi-class problems:
#     for such problems, please use MultiLabelConfusionMeter.
#     """

#     def __init__(self, k, normalized=False):
#         """
#         Args:
#             k (int): number of classes in the classification problem
#             normalized (boolean): Determines whether or not the confusion matrix
#                 is normalized or not
#         """
#         super(ConfusionMeter, self).__init__()
#         self.conf = np.ndarray((k, k), dtype=np.int32)
#         self.normalized = normalized
#         self.k = k
#         self.reset()

#     def reset(self):
#         self.conf.fill(0)

#     def add(self, predicted, target):
#         """
#         Computes the confusion matrix of K x K size where K is no of classes
#         Args:
#             predicted (tensor): Can be an N x K tensor of predicted scores obtained from
#                 the model for N examples and K classes or an N-tensor of
#                 integer values between 0 and K-1.
#             target (tensor): Can be a N-tensor of integer values assumed to be integer
#                 values between 0 and K-1 or N x K tensor, where targets are
#                 assumed to be provided as one-hot vectors

#         """
#         predicted = predicted.cpu().numpy()
#         target = target.cpu().numpy()

#         assert predicted.shape[0] == target.shape[0], \
#             'number of targets and predicted outputs do not match'

#         if np.ndim(predicted) != 1:
#             assert predicted.shape[1] == self.k, \
#                 'number of predictions does not match size of confusion matrix'
#             predicted = np.argmax(predicted, 1)
#         else:
#             assert (predicted.max() < self.k) and (predicted.min() >= 0), \
#                 'predicted values are not between 1 and k'

#         onehot_target = np.ndim(target) != 1
#         if onehot_target:
#             assert target.shape[1] == self.k, \
#                 'Onehot target does not match size of confusion matrix'
#             assert (target >= 0).all() and (target <= 1).all(), \
#                 'in one-hot encoding, target values should be 0 or 1'
#             assert (target.sum(1) == 1).all(), \
#                 'multi-label setting is not supported'
#             target = np.argmax(target, 1)
#         else:
#             assert (predicted.max() < self.k) and (predicted.min() >= 0), \
#                 'predicted values are not between 0 and k-1'

#         # hack for bincounting 2 arrays together
#         x = predicted + self.k * target
#         bincount_2d = np.bincount(x.astype(np.int32),
#                                   minlength=self.k ** 2)
#         assert bincount_2d.size == self.k ** 2
#         conf = bincount_2d.reshape((self.k, self.k))

#         self.conf += conf

#     def value(self):
#         """
#         Returns:
#             Confustion matrix of K rows and K columns, where rows corresponds
#             to ground-truth targets and columns corresponds to predicted
#             targets.
#         """
#         if self.normalized:
#             conf = self.conf.astype(np.float32)
#             return conf / conf.sum(1).clip(min=1e-12)[:, None]
#         else:
#             return self.conf
        
        

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        target = target[:,0]
        # Create vaiables
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input.cuda(async=True))
            target_var = torch.autograd.Variable(target.cuda(async=True))
        else:
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    confusion = ConfusionMeter(20)
    
    # Model on eval mode
    model.eval()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        target = target[:,0]
        # Create vaiables
        if torch.cuda.is_available():
            input_var = torch.autograd.Variable(input.cuda(async=True), volatile=True)
            target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)
        else:
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = torch.nn.functional.cross_entropy(output, target_var)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum() / batch_size, batch_size)
        losses.update(loss.data[0], batch_size)
        confusion.add(pred.squeeze(), target.cpu())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    print(confusion.value())
    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg 


def train(model, train_set, test_set, save, n_epochs=300, valid_size=5000,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Create train/valid split
    if valid_size:
        indices = torch.randperm(len(train_set))
        train_indices = indices[:len(indices) - valid_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_indices = indices[len(indices) - valid_size:]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Data loaders
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_size:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
        valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
        valid_loader = None

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            loader=valid_loader if valid_loader else test_loader,
            is_test=(not valid_loader)
        )

        # Determine if model is the best
        if valid_loader and valid_error < best_error:
            best_error = valid_error
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
        else:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)
