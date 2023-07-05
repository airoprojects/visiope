import os
import cv2
import sys
import math
import torch
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split



# This class rappresents the dataset
class Ai4MarsData(Dataset):

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

    def setPermuteX(self, perm):
        print(type(self.X))
        self.X = self.X.permute(perm[0], perm[1], perm[2], perm[3])

    def setPermuteY(self, perm):
        self.y = self.y.permute(perm[0], perm[1], perm[2], perm[3])

    def setDevice(self, device, which):

        if which == 0:
            self.X = self.X.to(device)

        else:
            self.y = self.y.to(device)

    def convertion(self, what):
        
        if what == 0:
            self.y = self.y.type(torch.DoubleTensor)
        
        else:
            self.X = self.X.type(torch.DoubleTensor)

    def resize(self, resize, interp=None):
        transform = transforms.Resize(resize,antialias=True)
        self.X = transform(self.X)
        self.y = transform(self.y)


# This function import the dataset as a two lists of nparray: X = images, y = labels 
class Ai4MarsImporter():

    def __init__(self, dataset_name='ai4mars-dataset-merged-0.1'):
        self.dataset = dataset_name
        ...


    def __call__(self, path='./', IN_COLAB=False, num_of_images=200): 
        print(f'This are the import parameters: \n \
              Path to the dataset: {path} \n \
              Colab Environment: {IN_COLAB} \n \
              Number of images to load: {num_of_images} \n'
              )

        if not IN_COLAB:
            is_here = os.path.exists(path + self.dataset)

            if is_here:
                print(f"You already have {self.dataset}")

            else:
                raise FileNotFoundError(self.dataset)

        else: # IN_COLAB

            fixed_path = '/content/drive/MyDrive/Dataset/'
            path = fixed_path

            from google.colab import drive
            drive.mount('/content/drive')

            # set up dataset directory path
            if not os.path.exists(fixed_path) : 
                os.makedirs(fixed_path)

            # Check if the dataset is already in Google Drive
            is_here = os.path.exists(fixed_path + self.dataset)

            if is_here:
                print(f"You already have {self.dataset}...")
                # os.chdir('/content/drive/MyDrive/Dataset/')

            else:
                # Check if the dataset is already in Google Drive but zipped
                import zipfile
                zip_file = fixed_path + self.dataset + '.zip'
                is_here_zipped  = os.path.exists(zip_file)
                
                if is_here_zipped :
                    print(f"You already have {self.dataset}, prepare to unzip it here: {fixed_path} ...")

                    # unzip dataset
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(fixed_path)  # extract in fixed path

                    # os.chdir('/content/drive/MyDrive/Dataset/')

                else:
                    print(f"You don't have {self.dataset}, prepare to download and unzip it {fixed_path} ...")

                    import gdown

                    # get url of zipped dataset
                    url = 'https://drive.google.com/uc?id=1eW9Ah9DDEY02CTHCrRYLGPmiGZvCTKK4'

                    # set up zip download location and start download
                    output = '/content/ai4mars-dataset-merged-0.1.zip'
                    gdown.download(url, output, quiet=False)

                    # unzip dataset
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(fixed_path)

                    #os.chdir('/content/drive/MyDrive/Dataset/')

        print(f"Unpacking images and lables from: {self.dataset}...")

        # Paths
        images = path + "ai4mars-dataset-merged-0.1/msl/images"
        label_train = path + "ai4mars-dataset-merged-0.1/msl/labels/train"
        label_test_1ag = path + "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min1-100agree"
        label_test_2ag = path + "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min2-100agree"
        label_test_3ag = path + "ai4mars-dataset-merged-0.1/msl/labels/test/masked-gold-min3-100agree"
        edr = images + "/edr"
        mxy = images + "/mxy"
        rng = images + "/rng-30m"

        # In this way we collect list of al files in the projects
        edr_files = os.listdir(edr)
        label_train_files = os.listdir(label_train)
        label_test_files_1 = os.listdir(label_test_1ag)
        label_test_files_2 = os.listdir(label_test_2ag)
        label_test_files_3 = os.listdir(label_test_3ag)

        X = []  # input
        y = []  # label train

        y1 = [] #test label 1 agree
        y2 = [] #test label 2 agree
        y3 = [] #test label 3 agree

        image_counter = 0 # this count how many images insert

        for label in label_test_files_1:
            path_label = os.path.join(label_test_1ag, label)
            img_arr = cv2.imread(path_label,0) 
            y1.append(img_arr)

        for label in label_test_files_2:
            path_label = os.path.join(label_test_2ag, label)
            img_arr = cv2.imread(path_label,0) 
            y2.append(img_arr)

        for label in label_test_files_3:
            path_label = os.path.join(label_test_3ag, label)
            img_arr = cv2.imread(path_label,0)
            y3.append(img_arr)

        for label in label_train_files:
            # Names of images match names of labels, except for the extension (JPG, png)
            img_name = label[:-4] + ".JPG" 

            if img_name in edr_files:
                img_path = os.path.join(edr, img_name) 
                img_arr = cv2.imread(img_path) 
                label_path = os.path.join(label_train, label)
                lab_arr = cv2.imread(label_path,0) # 0 mean read as greyscale image

                # Build nparray for inputs and outputs
                X.append(img_arr)
                y.append(lab_arr[:, :, np.newaxis])
            
                image_counter += 1  # this control how much images you want
                if image_counter == num_of_images: break
            
        print(f"Inputs len: {len(X)}")
        print(f"Labels len: {len(y)}")
        print("Done")
        return X, y

# This class perform some preprocessing including:
# Random Split
# Normalization
# Data Augmentation
class Ai4MarsProcessor():

    def __init__(self, X, y, transformation=None):
        self.X = X
        self.y = y
        self.transformation = transformation

    def __call__(self, percentages):
        X = self.X
        y = self.y
        transform = self.transformation

        # uncomment this to obtain the same split each experiment
        #random.seed(10)

        # assertions
        assert math.ceil(sum(percentages)) == 1.
        assert len(X) == len(y)

        dataset_len = len(X)
        subsets_lens = []
        total_values = 0

        # Computation of lenght of each subsets w.r.t. percentages  # 3-4 steps
        for percentage in percentages:
            value = math.floor(dataset_len * percentage)
            subsets_lens.append(value)
            total_values += value

        residuals = dataset_len - total_values

        # Redistributions of residuals due to floor function
        if residuals:

            for residual in range(1, residuals+1):                  
                subsets_lens[residual % len(percentages)] += 1

        subsets_indices = []
        random_indices = random.sample(range(dataset_len), dataset_len)
        start = 0

        # Random extrction of indices for each subsets
        for lens in subsets_lens:
            subsets_indices.append(random_indices[start:start+lens])
            start += lens

        subsets_X = [[] for i in range(len(subsets_indices))]
        subsets_y = [[] for i in range(len(subsets_indices))]
        i = 0

        # Splitting of dataset into subsets
        for indices in subsets_indices:

            for index in indices:
                subsets_X[i].append(X[index])
                subsets_y[i].append(y[index])
            i += 1

        # convertion to np array and Normalization                   
        for i in range(len(subsets_X)):
            subsets_X[i] = np.asanyarray(subsets_X[i], dtype= np.float32) / 255
            subsets_y[i] = np.array(subsets_y[i], dtype= np.int64)
            subsets_y[i][subsets_y[i] == 255] = 4

        # Convertion to torch tensors
        for i in range(len(subsets_X)):                             
            subsets_X[i] = torch.from_numpy(subsets_X[i]).permute(0,3,2,1)
            subsets_y[i] = torch.from_numpy(subsets_y[i]).permute(0,3,2,1)

        augmentation_X = []
        augmentation_y = []

        # Data Augmentation
        if transform:

            for tensor_X, tensor_y in zip(subsets_X[0], subsets_y[0]):
                # Save the state of the tensors
                state = torch.get_rng_state()
                augmentation_X.append(transform(tensor_X))

                # To then apply the same transformation
                torch.set_rng_state(state)
                augmentation_y.append(transform(tensor_y))

            augmentation_X = torch.stack(augmentation_X, dim=0)
            augmented_set_X = torch.cat((subsets_X[0], augmentation_X[:100]),0)

            augmentation_y = torch.stack(augmentation_y, dim=0)
            augmented_set_y = torch.cat((subsets_y[0], augmentation_y[:100]),0)

        datasets = []

        # Creation of datasets
        for i in range(len(subsets_X)):

            if transform and i == 1:
                datasets.append(Ai4MarsData(augmented_set_X, augmented_set_y))
            
            else:
                datasets.append(Ai4MarsData(subsets_X[i], subsets_y[i]))

        return datasets
    
if __name__ == '__main__':
    ...