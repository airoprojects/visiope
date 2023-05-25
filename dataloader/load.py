import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset, random_split



#This class rappresents the dataset 
class Ai4MarsData(Dataset):
    #X tensor (torch) -> images
    #y tensor (torch) -> labels

    def __init__(self, X, y,transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        image = self.X[index]
        label = self.y[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    

    '''
    TO DO
  
    def resize(self,resize,interp=None):
        if interp:
            k = interp
        else:
            k = cv2.INTER_NEAREST
        for image in self.X:
            image = 
        self.X = cv2.resize(img_arr, dsize=(size, size),interpolation=k)
        self.y = cv2.resize(img_arr, dsize=(size, size),interpolation=k)

    '''
    
    def setPermuteX(self,perm):   
        print(type(self.X)) 
        self.X = self.X.permute(perm[0],perm[1],perm[2],perm[3])
    
    def setPermuteY(self,perm):    
        self.y = self.y.permute(perm[0],perm[1],perm[2],perm[3])

    
    def setDevice(self,device,which):
        if which==0:
            self.X = self.X.to(device)
        else:
            self.y = self.y.to(device)

    def convertion(self,what):
        if(what==0):
            self.y = self.y.type(torch.DoubleTensor)
        else:
            self.X = self.X.type(torch.DoubleTensor)
        
    
    #this function return 3 dataloader (train,test,validation) splitted from self 
    #percentage -> give percentage of train size, the rest of percentage is given divided the residual part
    #sizeBatch -> determine the size of batch
    def splitLoader(self,percentage,sizeBatch):
        dataset = self
        ratio = percentage/100

        #setup variables
        d_size = len(self)
        train_size = int(ratio*d_size)
        test_size = int((d_size - train_size)/2)
        validation_size = test_size

        #split
        train_dataset, test_dataset, validation_dataset = random_split(dataset,[train_size,test_size,validation_size])



       
        

        print(type(train_dataset))

        #create other loaders
        train_loader = DataLoader(train_dataset,batch_size=sizeBatch)
        test_loader = DataLoader(test_dataset,batch_size=sizeBatch)
        validation_loader = DataLoader(validation_dataset,batch_size=sizeBatch)



        return train_loader,test_loader,validation_loader
    