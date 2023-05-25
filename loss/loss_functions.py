"""  This script implements different loss function, both standard and custom, used 
to train MER-Segmentation to perform semantic segmentation over martian terrain  
images. """

import torch.nn as nn
import numpy as np

class LossClass:

    @classmethod
    def __show__():
        print("\n   Standard Cross Entropy \n")

    # Standard cross entropy ... useless
    def std_cross_entropy(label, prediction):
        criteria = nn.CrossEntropyLoss()
        return criteria(label, prediction)

    # Soft Dice loss ... to be delivered
