# General imports

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Custom Imports

from git import Repo

# Initialize the Git repository object
repo = Repo(".", search_parent_directories=True)

# Get the root directory of the Git project
root_dir = repo.git.rev_parse("--show-toplevel")

# Add custom modules to path
custom_modules_path = root_dir  + '/tools/'
sys.path.insert(0, custom_modules_path)


# Import Loader
from data.utils import Ai4MarsDownload, Ai4MarsImporter

# Buld up a dataset with n torch tensor from ai4marsdataset

# Insert here your local path to the dataset (temporary on airodrive)
#raise Exception('Remove this line and inset the path to the dataset below')
data_path = '/home/leeoos/Desktop/'

# Local path to save the dataset
save_path = root_dir + '/datasetup/dataset/'

# Import data as Ai4MarsDataset
Ai4MarsDownload()(PATH=data_path)
importer = Ai4MarsImporter()
X_, y_, _ = importer(PATH=data_path, NUM_IMAGES=200, SAVE_PATH=save_path, checkpoint=0)

# Extend the dataset
extend = True
if extend:
    for i in range(3):
      X_, y_, _ = importer(PATH=data_path, NUM_IMAGES=200, SAVE_PATH=save_path, checkpoint=_)
      
      del X_
      del y_

