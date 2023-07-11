"""  This script implements different loss functions, both standard and custom, used 
to train MER-Segmentation to perform semantic segmentation over martian terrain  
images. """

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Soft Dice Loss ... to review  
class GDL(nn.Module):

    # Template initialization of nn Module to create callable objects
    def __init__(self):
        super(GDL, self).__init__()
        self.epsilon = epsilon=1e-6

    # Actual implementation of Generalized Dice Loss
    def forward(self, label, prediction):
        assert label.size() == prediction.size(), "'prediction' and 'target' must have the same shape"

        # Flatten all dimentions
        label = torch.flatten(label)
        prediction = torch.flatten(prediction)

        w_l = label.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (prediction * label).sum(-1)
        print(intersect.shape)
        intersect = intersect * w_l

        denominator = (prediction + label).sum(-1)
        print(denominator)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 1 - (2 * (intersect.sum() / denominator.sum()))
    
# ChatGPT implementation. Highly probable that is it not correct
class GPTFocalLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2.0):
        super(GPTFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Compute the focal loss
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Compute the mean loss over the batch
        loss = torch.mean(focal_loss)

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class SoftDice(nn.Module):

    def forward(y_true, y_pred, epsilon=1e-6): 
        ''' 
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.
    
        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
            epsilon: Used for numerical stability to avoid divide by zero errors
        
        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation 
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
            
            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        '''
        
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * np.sum(y_pred * y_true, axes)
        denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
        
        return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch
        # thanks @mfernezir for catching a bug in an earlier version of this implementation!