""" 
This script implements different loss function, both standard and custom, used 
to train MER-Segmentation to perform semantic segmentation over martian terrain  
images.
"""

import torch.nn as nn
import numpy as np

# Standard cross entropy 
def std_cross_entropy(label, prediction):
    criteria = nn.CrossEntropyLoss()
    return criteria(label, prediction)

# Soft Dice loss
def soft_dice(y_true, y_pred, epsilon=1e-6): 
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and 
    number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c 
        channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors

    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image 
        Segmentation https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from 
        https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    # average over classes and batch
    return 1 - np.mean((numerator + epsilon) / (denominator + epsilon)) 
    # thanks @mfernezir for catching a bug in an earlier version of this 
    #implementation!
