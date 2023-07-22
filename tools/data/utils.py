""" This is a custom module to download, set up and process the ai4mars-dataset-merged-0.1
to perform semantic segmentation over martian terrain images."""

import os
import cv2
import math
import torch
import random
import numpy as np
from typing import Any
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

#===============================================================#
# Custom dataset for semantic segmentetion on Mars terrain images
class Ai4MarsDataset(Dataset):

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

    def set_permute(self, perm):
        self.X = self.X.permute(perm[0], perm[1], perm[2], perm[3])
        self.y = self.y.permute(perm[0], perm[1], perm[2], perm[3])
        

    def set_device(self, device):
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def conversion(self, new_type='f'):

        if new_type == 'f':
            self.X = self.X.type(torch.float32)
            self.y = self.y.type(torch.float32)
        
        elif new_type == 'd':
            self.X = self.X.type(torch.float64)
            self.y = self.y.type(torch.float64)
        
        else:
            raise Exception('Invalid type')

    def resize(self, resize, interp=None):
        transform = transforms.Resize(resize,antialias=True)
        self.X = transform(self.X)
        self.y = transform(self.y)

    def set_grad(self):
        self.X.requires_grad = True

#===============================================================#
# Download the dataset 
class Ai4MarsDownload():

    def __init__(self) -> None:
        pass

    def __call__(self, PATH:str='./'): 

        import sys
        COLAB = 'google.colab' in sys.modules
        LOCAL = not(COLAB)

        DATASET = 'ai4mars-dataset-merged-0.1'

        print(f"Downloader script for {DATASET}")
        
        # Check if the dataset is already in PATH
        is_here = os.path.exists(PATH + DATASET)

        if LOCAL:

            if is_here:
                print(f"You already have {DATASET}")
                pass

            else:
                print(f'{DATASET} not found, prepare to download it here: {PATH}')
                import gdown

                # get url of zipped dataset
                url = 'https://drive.google.com/uc?id=1eW9Ah9DDEY02CTHCrRYLGPmiGZvCTKK4'

                # set up zip download location and start download
                output = PATH + 'ai4mars-dataset-merged-0.1.zip'
                gdown.download(url, output, quiet=False)

        elif COLAB:

            # Mount Google Drive to be on the safe side 
            from google.colab import drive
            drive.mount('/content/drive')

            # set up dataset directory path
            if not os.path.exists(PATH) : 
                os.makedirs(PATH)

            if is_here:
                print(f"You already have {DATASET}...")

            else:

                # Check if the dataset is already in Google Drive but zipped
                import zipfile
                from tqdm import tqdm

                zip_file = PATH + DATASET + '.zip'
                is_here_zipped  = os.path.exists(zip_file)
                
                if is_here_zipped :
                    print(f"You already have {DATASET}, prepare to unzip it here: {PATH} ...")

                    # unzip dataset
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                            try:
                                zip_ref.extract(member, PATH)
                            except zipfile.error as e:
                                pass

                else:
                    print(f"You don't have {DATASET}, prepare to download and unzip it in {PATH} ...")

                    import gdown

                    # get url of zipped dataset
                    url = 'https://drive.google.com/uc?id=1eW9Ah9DDEY02CTHCrRYLGPmiGZvCTKK4'

                    # set up zip download location and start download
                    output = '/content/ai4mars-dataset-merged-0.1.zip'
                    gdown.download(url, output, quiet=False)

                    # unzip dataset
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        for member in tqdm(zip_ref.infolist(), desc='Extracting '):
                            try:
                                zip_ref.extract(member, PATH)
                            except zipfile.error as e:
                                pass
        print("Done\n")
        return PATH

#===============================================================#
# Import the dataset as torch array
class Ai4MarsImporter():
        
    def __call__(self, PATH:str='./', NUM_IMAGES=200, SAVE_PATH:str=None, SIZE:int=128, checkpoint:int=None): 

        if checkpoint: NUM_IMAGES += checkpoint

        # Allow to process all the images in the dataset
        if NUM_IMAGES == 'all': NUM_IMAGES = 16064

        if NUM_IMAGES > 16064 : 
            raise Exception(f"Trying to import too many images: {NUM_IMAGES}. Max number: 16000")

        DATASET = 'ai4mars-dataset-merged-0.1'

        print(f"Import parameters: \n \
              Dataset: {DATASET} \n \
              Path to the dataset: {PATH} \n \
              Number of images to load: {NUM_IMAGES} \n \
              Saving path for X and y: {SAVE_PATH}"
              )
        
        # Unpacking Phase 
        print(f"Unpacking images and lables from: {DATASET} ...")

        # Images and labels paths
        images = PATH + "ai4mars-dataset-merged-0.1/msl/images"
        label_train = PATH + "ai4mars-dataset-merged-0.1/msl/labels/train"
        edr = images + "/edr"

        # In this way we collect list of al files in the projects
        edr_files = os.listdir(edr)
        label_train_files = os.listdir(label_train)

        X = []  # input
        y = []  # label train

        image_counter = 0 # this count how many images insert
        
        for label in label_train_files:

            # This allow to load data after a given checkpoint
            if checkpoint and (image_counter < checkpoint): 
                image_counter += 1 
                pass

            else: 
                image_counter += 1

                # Names of images match names of labels, except for the extension (JPG, png)
                img_name = label[:-4] + ".JPG" 

                if img_name in edr_files:
                    img_path = os.path.join(edr, img_name) 
                    img_arr = cv2.imread(img_path) 
                    label_path = os.path.join(label_train, label)
                    lab_arr = cv2.imread(label_path,0) # 0 mean read as greyscale image

                    # Build torch tensors
                    x_t = torch.from_numpy(img_arr) / 255 # normalization
                    y_t = torch.from_numpy(lab_arr[:, :, np.newaxis])
                    y_t[y_t == 255] = 4 # reassigment for background

                    if SIZE:
                        transform = transforms.Resize(SIZE,antialias=True)
                        x_t = x_t.permute(2,1,0)
                        x_t = transform(x_t)
                        y_t = y_t.permute(2,1,0)
                        y_t = transform(y_t)
                        
                    X.append(x_t)
                    y.append(y_t)

                    # free up some memory
                    del img_arr
                    del lab_arr
            
            if image_counter == NUM_IMAGES: break
            
        print(f"Inputs len: {len(X)}")
        print(f"Labels len: {len(y)}")

        print("Converting inputs and labels into torch tensors ...")
        X = torch.stack(X, dim=0) # 3 x SIZE x SIZE
        y = torch.stack(y, dim=0) # 1 x SIZE x SIZE

        if SAVE_PATH:
            print(f"{DATASET} will be saved in: {SAVE_PATH}")
            # torch.save(X, SAVE_PATH + 'X.pt')
            # torch.save(y, SAVE_PATH + 'y.pt')

            if not os.path.isfile(SAVE_PATH + 'dataset.pt'):
                torch.save((X, y), SAVE_PATH + 'dataset.pt')

            else:
                print(f"A version of {DATASET} already exist, appending new data ...")
                OLD_X, OLD_y = torch.load(SAVE_PATH + 'dataset.pt')
        
                NEW_X = torch.cat((OLD_X, X), dim=0)
                NEW_y = torch.cat((OLD_y, y), dim=0)
                torch.save((NEW_X, NEW_y ), SAVE_PATH + 'dataset.pt')

                del OLD_X
                del OLD_y
                del NEW_X
                del NEW_y

        print("Done\n")
        return X, y, image_counter

#===============================================================#
# Random Split and Data Augmentation
class Ai4MarsSplitter():

    def __init__(self) -> None:
        self.augmentation_size = 100  # hyperparamto decide the size of the augmentation
        self.info = {}
        pass

    def __call__(self, X, y, percentages:list=None, transform=None, SAVE_PATH:str=None, SIZE:int=None):

        import sys
        COLAB = 'google.colab' in sys.modules
        LOCAL = not(COLAB)
        
        DATASET = 'ai4mars-dataset-merged-0.1'

        # Saving data parameters
        self.info['dataset'] = DATASET
        self.info['percentages'] = str(percentages)
        self.info['transform'] = transform

        if SIZE:
            self.info['size'] = str(SIZE)
        
        else:
            self.info['size'] = str(X.shape[-2])

        print(f"Splitting parameters: \n \
            Dataset: {DATASET} \n \
            Colab environment: {COLAB} \n \
            Split percentages: {percentages} \n \
            Transformation: {transform} \n \
            Svaving path: {SAVE_PATH} \n \
            New image size: {self.info['size']}")
        

        # if COLAB:
        #     answ = str(input("Do you want to perform a lighter processing for the data? ")).lower()

        #     if answ in ['yes', 'y', 'si', 's']:

        #         from torch.utils.data import random_split
        #         dataset = Ai4MarsDataset(X, y)
        #         if SIZE: dataset.resize(SIZE)

        #         return random_split(dataset, percentages)

        datasets = []

        # If no percentages are given the data are not splitted and only one
        # Dataset object is returned. In this case no transformations are applied
        if not percentages:
            datasets.append(Ai4MarsDataset(X, y))
            return datasets[0]

        # uncomment this to obtain the same split each experiment
        #random.seed(10)

        # assertions
        assert math.ceil(sum(percentages)) == 1. , 'Percentages should sum to 1'
        assert len(X) == len(y), 'Len of Inputs and Labels must be the same'

        print("Extrapolation of random inices ...")

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

        # Random extraction of indices for each subsets
        for lens in subsets_lens:
            subsets_indices.append(random_indices[start:start+lens])
            start += lens

        subsets_X = [[] for i in range(len(subsets_indices))]
        subsets_y = [[] for i in range(len(subsets_indices))]
        i = 0

        print("Splitting in progress ...")

        # Splitting of dataset into subsets
        for indices in subsets_indices:

            for index in indices:
                subsets_X[i].append(X[index])
                subsets_y[i].append(y[index])
            i += 1

        # Convertion to torch tensors - x C x H x W 
        for i in range(len(subsets_X)):                             
            subsets_X[i] = torch.stack(subsets_X[i], dim=0) #.permute(0,1,3,2).permute(0,2,1,3)
            subsets_y[i] = torch.stack(subsets_y[i], dim=0) #.permute(0,1,3,2).permute(0,2,1,3)

        augmentation_X = []
        augmentation_y = []

        # Data Augmentation
        if transform:
            print("Data Augmentation is a memory consuming operation ...")

            for tensor_X, tensor_y in zip(subsets_X[0], subsets_y[0]):

                # Save the state of the tensors
                state = torch.get_rng_state()
                tensor_T_X  = transform(tensor_X)
                augmentation_X.append(tensor_T_X)

                # To then apply the same transformation
                torch.set_rng_state(state)
                tensor_T_y  = transform(tensor_y)
                augmentation_y.append(transform(tensor_T_y))

                tensor_T_X.detach()
                tensor_T_y.detach()
                del tensor_T_X
                del tensor_T_y

            augmentation_X = torch.stack(augmentation_X, dim=0)
            augmented_set_X = torch.cat((subsets_X[0], augmentation_X[:self.augmentation_size]),0)

            augmentation_y = torch.stack(augmentation_y, dim=0)
            augmented_set_y = torch.cat((subsets_y[0], augmentation_y[:self.augmentation_size]),0)

        # Datasets
        for i in range(len(subsets_X)):

            # Add images only on train set: 
            # this assume the split to be done with train at position 0
            if transform and i == 0:
                datasets.append(Ai4MarsDataset(augmented_set_X, augmented_set_y))
            
            else:
                datasets.append(Ai4MarsDataset(subsets_X[i], subsets_y[i]))

        if SIZE:
            print(f"Resizing the {DATASET} images at size: {SIZE} ...")
            print("Resizing is a memory consuming operation ...")

            for dataset in datasets:
                dataset.resize(SIZE)

        if SAVE_PATH:
            print(f"The Ai4MarsDatasets will be saved here: {SAVE_PATH}")

            import os 
            if not os.path.exists(SAVE_PATH) : 
                os.makedirs(SAVE_PATH)
            torch.save(datasets, SAVE_PATH + 'splitted_dataset.pt')
            
        torch.save(self.info, './.info.pt')

        print("Done \n")
        return datasets

#===============================================================#
# Custom dataloader   
class Ai4MarsDataLoader():

    def __init__(self) -> None:
        pass

    def __call__(self, datasets=None, batch_sizes:list=[32,32,32], SIZE:int=None, 
                 TYPE:str='f', SAVE_PATH:str=None) -> Any:
        
        assert len(datasets) == len(batch_sizes), 'You must provide a batch size for each Dataloader'

        print("Building Dataloaders")
        
        dataloaders = []

        # Resize images and labels to fit in RAM
        if SIZE:
            for dataset in datasets:
                dataset.resize(SIZE)
        
        # Convert dataset to float tensors to be on the safe side
        for b, dataset in enumerate(datasets):
            dataset.conversion(TYPE)
            dataset.set_grad()  # Enable Gradient to be on the safe side
            dataloaders.append(DataLoader(dataset, batch_sizes[b], shuffle=True))


        if SAVE_PATH:
            print(f"The Ai4MarsDataloaders will be saved here: {SAVE_PATH}")

            import os 
            if not os.path.exists(SAVE_PATH) : 
                os.makedirs(SAVE_PATH)
            torch.save(dataloaders, SAVE_PATH + 'dataloaders' + '.pt')

        print("Done \n")
        return dataloaders
    
if __name__ == '__main__':
    pass