"""  This script implements different loss functions, both standard and custom, used 
to train MER-Segmentation to perform semantic segmentation over martian terrain  
images. """

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

        
class GDL(nn.Module):
    # Soft Dice Loss ... to review  

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
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
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

