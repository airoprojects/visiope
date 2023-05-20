import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset, random_split



#This class rappresents the dataset 
class Ai4MarsData(Dataset):
    #X tensor (torch) -> images
    #y tensor (torch) -> labels

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[label]

        if self.transform:
            image = self.transform(image)

        return image, label
    def splitLoader(self,percentage,batch_size_train,batch_size_test):
        dataset = self
        ratio = percentage/100

        #setup variables
        d_size = len(self)
        train_size = int(ratio*d_size)
        test_size = d_size - train_size

        #split
        train_dataset, test_dataset = random_split(dataset,[train_size,test_size])

        #create other loaders
        train_loader = DataLoader(train_dataset,batch_size=batch_size_train)
        test_loader = DataLoader(test_dataset,batch_size=batch_size_test)
        return train_loader,test_loader


