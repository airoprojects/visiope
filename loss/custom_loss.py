"""  This script implements different loss functions, both standard and custom, used 
to train MER-Segmentation to perform semantic segmentation over martian terrain  
images. """

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

        

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