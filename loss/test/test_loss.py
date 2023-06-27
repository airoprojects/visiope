from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
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
# print(label.shape)
# print(label.type())
label = label.type(torch.DoubleTensor)
print(label.shape)
# print(label.type())
# print(label)


prediction = torch.load("test/fake_prediction_lable.pt")
print(prediction.shape)
#print(prediction.type())


# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax1.imshow(label)
# ax2 = fig.add_subplot(2,2,2)
# ax2.imshow(prediction)
# plt.show()

# does not work
#soft_dice(label, prediction)

testing_losses =  []
testing_losses.append(custom_loss.GDL())
testing_losses.append(nn.CrossEntropyLoss())
for current_loss in testing_losses:
    loss_fn = current_loss
    loss_value = current_loss(label, prediction)
    print(f"Current loss: {current_loss}, value: {loss_value}")




# Example of target with class indices
# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(input)
# print(target)
# output = loss(input,input)
# print(output)




