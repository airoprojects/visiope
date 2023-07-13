## This is the header that should be put in main to import the dataset correctly

**Cell One**

```
# Imports, , Dataset, Splitting, Dataloader, Training and Loss functions

import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Custom Imports
COLAB = 'google.colab' in sys.modules
LOCAL = not COLAB

if COLAB

    # Clone visiope repo on runtime env
    !git clone https://github.com/airoprojects/visiope.git

    # Add custom modules to path
    custom_modules_path = '/content/visiope/tools/'
    sys.path.insert(0, custom_modules_path)

elif LOCAL:

    from git import Repo

    # Initialize the Git repository object
    repo = Repo(".", search_parent_directories=True)

    # Get the root directory of the Git project
    root_dir = repo.git.rev_parse("--show-toplevel")

    # Add custom modules to path
    custom_modules_path = root_dir  + '/tools/'
    sys.path.insert(0, custom_modules_path)

else:
    raise Exception("Unknown Environment")


# Import Loader
from data.utils import Ai4MarsDownload, Ai4MarsSplitter

# Import Loss
import loss.loss

# Import Trainer
import trainer.trainer

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Cell Two**

```
# Dataloader

LOAD = True

if LOAD:

    if not(os.path.exists('/content/dataset/'):

        import gdown

        # get url of torch dataset (temporarerly my drive)
        url_X_200 = 'https://drive.google.com/uc?id=1HLAQgjZbGa3lMzdyIvgykCiUJ6P4OaEX'
        url_y_200 = 'https://drive.google.com/uc?id=1Ue74LEe0WlEnFxRcIl19WyvXR1EyYIsS'

        url_X_1000 = 'https://drive.google.com/uc?id=1zvXKK1qc2PNbAMyq0XWfqrhAWB5kO3yO'
        url_y_1000 = 'https://drive.google.com/uc?id=1gmnWAGjZJYt-VIpWRK_cEOXQfFfuFAwe'

        # download np dataset on runtime env
        data_path = '/content/dataset/'
        gdown.download(url_X_200, data_path, quiet=False)
        gdown.download(url_y_200, data_path, quiet=False)

    train_set = torch.load("/content/dataset/train.pt")
    val_set = torch.load("/content/dataset/val.pt")
    test_set = torch.load("/content/dataset/test.pt")
        
elif LOCAL or not LOAD:

    # Insert here your local path to the dataset (temporary)
    data_path = input("Path to Dataset: ")

    # Insert here the number of images you want to download
    num_images = int(input("Number of images (max 1000): "))

    if num_images > 1000 : raise Exception("Trying to import too many images")

    # Import data as Ai4MarsDataset
    importer = Ai4MarsDownload()
    X, y = importer(PATH=data_path, NUM_IMAGES=100)

    # Split the dataset
    splitter = Ai4MarsSplitter()
    train_set, test_set, val_set = splitter(X, y, [0.7, 0.2, 0.1])

# Resize images and labels to fit in RAM
train_set.resize(64)
test_set.resize(64)
val_set.resize(64)

# Convert dataset to float tensors to be on the safe side
train_set.conversion('f')
test_set.conversion('f')
val_set.conversion('f')

# Enable Gradient to be on the safe side
train_set.set_grad()
test_set.set_grad()
val_set.set_grad()

# Making of dataloader
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

```