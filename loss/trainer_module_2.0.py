""" Trainer module for MER-Segmentation 2.0 """

import sys
import torch
from pathlib import Path
from datetime import datetime

# Check if the script is runned in local or on colab
IN_COLAB = 'google.colab' in sys.modules

# Path object for the current file
current_file = Path(__file__).parent
# Setup path for custom imports in local and colab
if IN_COLAB: sys.path.insert(0, '/content/drive/MyDrive/Github/visiope/loss')
else: sys.path.insert(0, current_file)
from custom_loss import *
import custom_loss