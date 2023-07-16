# General imports

import os
import sys
import torch
import numpy as np
from git import Repo
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Custom Imports

# Initialize the Git repository object
repo = Repo(".", search_parent_directories=True)

# Get the root directory of the Git project
root_dir = repo.git.rev_parse("--show-toplevel")

# Add custom modules to path
custom_modules_path = root_dir  + '/tools/'
sys.path.insert(0, custom_modules_path)


# Import Loader
from data.utils import Ai4MarsDownload, Ai4MarsSplitter, Ai4MarsDataLoader

# Dataloader

# Insert here your local path to the dataset (temporary on airodrive)
#raise Exception('Remove this line and inset the path to the dataset below')
data_path = '/home/leeoos/Desktop/'

# Local path to save the dataset
save_path = root_dir + '/datasetup/dataset/'

# Import data as Ai4MarsDataset
importer = Ai4MarsDownload()
X, y = importer(PATH=data_path, NUM_IMAGES=500)

# Uncomment the following lines to apply transformations to the dataset
transform = transforms.RandomChoice([
    transforms.RandomRotation(90)])

# Split the dataset
splitter = Ai4MarsSplitter()
train_set, test_set, val_set = splitter(X, y, [0.7, 0.2, 0.1], transform=transform,
                                        SAVE_PATH=save_path, SIZE=128)

# Build Ai4MarsDataloader
loader = Ai4MarsDataLoader()
train_loader, test_loader, val_loader = loader([train_set, test_set, val_set], [32, 16, 16], SIZE=128)
