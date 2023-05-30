from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pickle
import torch

# Setup path for custom imports
import sys
sys.path.insert(0, '/content/drive/MyDrive/Github/visiope/loss')
sys.path.insert(0, '../loss')
from custom_loss import *
import custom_loss

# Train the model
train_loader = []


label = torch.load("test/true_lable.pt")
print(label.shape)
print(label.type())
label = label.type(torch.DoubleTensor)
print(label.type())
print(label)
"""plt.imshow(label)
plt.show()"""


prediction = torch.load("test/fake_prediction_lable.pt")
print(prediction.shape)
print(prediction.type())
"""plt.imshow(prediction)
plt.show()"""

# does not work
#soft_dice(label, prediction)
loss = LossClass

criteria = nn.CrossEntropyLoss()
print(criteria)
print(criteria(label, prediction))

criteria2 = custom_loss.GDL()
print(criteria2)
print(criteria2(label, prediction))




#loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




