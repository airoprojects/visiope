from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pickle
import torch
import sys
sys.path.insert(0, '/content/drive/MyDrive/Github/visiope/loss')
from loss_fn import *


"""
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
        label = self.y[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    

with open('/content/drive/MyDrive/Dataset/data_loader.pkl', 'rb') as f:
    data_loader = pickle.load(f)


items = data_loader['dataloader'].dataset.__getitem__(0)

label = items[1]

torch.save(label, 'test/true_lable.pt')
"""

label = torch.load("test/true_lable.pt")
print(label.shape)
print(label.type())
label = label.type(torch.DoubleTensor)
print(label.type())
plt.imshow(label)
plt.show()


prediction = torch.load("test/fake_prediction_lable.pt")
print(prediction.shape)
print(prediction.type())
plt.imshow(prediction)
plt.show()

# does not work
#SoftDiceLoss(label, prediction)

criteria_result = CrossEntropyStd(label, prediction)
print(criteria_result)




