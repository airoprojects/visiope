"""  This script implements different loss functions, both standard and custom, used 
to train MER-Segmentation to perform semantic segmentation over martian terrain  
images. """

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

class Ai4MarsCrossEntropy(nn.Module):
    def __init__(self, 
                 ignore_label: int = 4, 
                 weight: Tensor = None) -> None:
        super().__init__()
        self.ignore_index = ignore_label
        self.weight = weight

    def forward(self, 
                inputs: Tensor, 
                target: Tensor) -> Tensor:
        
        # Check for inputs errors
        if not torch.is_tensor(inputs):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(inputs)))
        
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect B x H x W x C. Got: {}".format(inputs.shape))
        
        # if not inputs.shape[1:2] == target.shape[1:2]:
        #     raise ValueError("input and target shapes must be the same. Got: {}"
        #                      .format(inputs.shape, targets.shape))

        if not inputs.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {})"
                             .format((inputs.device, target.device)))
        
        # Cross entropy needs inputs in original shape B x C x H x W but we have B x H x W x C
        rev_inputs = inputs.permute(0, 1, 3, 2).permute(0, 2, 1, 3)
        
        criterion = nn.CrossEntropyLoss(self.weight) #, self.ignore_index=ignore_label)
        
        return criterion(rev_inputs, target)


class Ai4MarsDiceLoss(nn.Module):

    def __init__(self) -> None:
        super(Ai4MarsDiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(
            self,
            inputs: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:

        # Check for inputs errors
        if not torch.is_tensor(inputs):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(inputs)))
        
        if not len(inputs.shape) == 4:
            raise ValueError("Invalid input shape, we expect B x H x W x C. Got: {}".format(inputs.shape))
        
        # if not inputs.shape[1:2] == target.shape[1:2]:
        #     raise ValueError("input and target shapes must be the same. Got: {}"
        #                      .format(inputs.shape, targets.shape))

        if not inputs.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {})"
                             .format((inputs.device, target.device)))

        # Assume input shape as B x H x W x C
        num_classes = inputs.shape[-1]

        # compute softmax over the classes axis
        inputs_soft = F.softmax(inputs, dim=-1)

        # Create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=num_classes)

        # Compute intersection and cardinality of inputs and target intersection and cardinality
        intersection = torch.sum(inputs_soft.reshape(-1) * target_one_hot.reshape(-1), -1)
        cardinality = torch.sum(inputs_soft.reshape(-1) + target_one_hot.reshape(-1), -1)

        dice_score = 2. * intersection / (cardinality + self.eps)

        return 1. - dice_score  # dice loss is 1 - dice score
    
if __name__ == '__main__':
    pass