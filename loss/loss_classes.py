print("I work")
import torch.nn as nn

class LossClass:
    """ this class implements different loss function 
    both standard and classic, for semantic segmentation task """

    def standard_cross_entropy(outputs, labels):
        nn.CrossEntropyLoss()
