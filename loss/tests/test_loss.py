from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import cv2
import sys
import torch
import torch.nn as nn

IN_COLAB = 'google.colab' in sys.modules

if not IN_COLAB:

    from git import Repo

    # Initialize the Git repository object
    repo = Repo(".", search_parent_directories=True)

    # Get the root directory of the Git project
    root_dir = repo.git.rev_parse("--show-toplevel")

    from pathlib import Path

    # Set up path for custom loss modules
    loss_module = root_dir + '/loss/'
    sys.path.insert(0, loss_module)

    # Set up path for custom loss modules
    trainer_module = root_dir + '/trainer/'
    sys.path.insert(0, trainer_module)

    # Insert here your local path to the dataset
    data_path = '/home/leeoos/Desktop/'

else: 
    
    from google.colab import drive
    drive.mount('/content/drive')

    # On Colab the path to the module is fixed once you have 
    # correttly set up the project with gitsetup.ipynb 
    fixed_path_loss = '/content/drive/MyDrive/Github/visiope/loss/'
    sys.path.insert(0, fixed_path_loss)

    # Insert here the path to the dataset on your drive
    data_path = '/content/drive/MyDrive/Dataset/'

import asloss

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
testing_losses.append(asloss.GDL())
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




