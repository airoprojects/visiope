import os
import cv2
import math
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# This class rappresents the dataset
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


# This class import the dataset as a two lists of nparray: X = images, y = labels 
class Ai4MarsDownload():

    def __call__(self, PATH='./', NUM_IMAGES=200, SAVE_PATH=None): 

        DATASET = 'ai4mars-dataset-merged-0.1'

        import sys
        COLAB = 'google.colab' in sys.modules
        LOCAL = not(COLAB)

        # Downloading Phase
        print(f'This are the import parameters: \n \
              Dataset: {DATASET}
              Path to the dataset: {PATH} \n \
              Colab Environment: {COLAB} \n \
              Number of images to load: {NUM_IMAGES} \n \
              Saving path for X and y: {SAVE_PATH}'
              )
        
        # Allow to process all the images in the dataset
        if NUM_IMAGES== 'all': NUM_IMAGES = 16064

        # # Check if the dataset is already in PATH
        is_here = os.path.exists(PATH + DATASET)

        if LOCAL:

            if is_here:
                print(f"You already have {DATASET}")
                pass

            else:
                raise FileNotFoundError(DATASET)

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
                                zip_ref.extract(member, path)
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
                                zip_ref.extract(member, path)
                            except zipfile.error as e:
                                pass
        
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

            # Names of images match names of labels, except for the extension (JPG, png)
            img_name = label[:-4] + ".JPG" 

            if img_name in edr_files:
                img_path = os.path.join(edr, img_name) 
                img_arr = cv2.imread(img_path) 
                label_path = os.path.join(label_train, label)
                lab_arr = cv2.imread(label_path,0) # 0 mean read as greyscale image

                # Build torch tensors
                x_t = torch.from_numpy(img_arr) / 255 # normalization
                X.append(x_t)

                y_t = torch.from_numpy(lab_arr[:, :, np.newaxis])
                y_t[y_t == 255] = 4 # reassigment of background
                y.append(y_t)

                # free up some memory
                del img_arr
                del lab_arr
            
                image_counter += 1  # this control how much images you want
                if image_counter == num_of_images: break
            
        print(f"Inputs len: {len(X)}")
        print(f"Labels len: {len(y)}")

        print("Converting inputs and labels in torch tensors ...")
        X = torch.stack(X, dim=0)
        y = torch.stack(y, dim=0)
        print("Done")

        if SAVE_PATH:
            print(f"Your dataset will be saved in two different files in: {SAVE_PATH}")
            torch.save(X, SAVE_PATH + 'X.pt')
            torch.save(y, SAVE_PATH + 'y.pt')

        return X, y

# This class perform Random Split and Data Augmentation
class Ai4MarsProcessor():

    def __init__(self):
        ...

    def __call__(self, X, y, percentages=None, transform=None):

        datasets = []

        # If no percentual are given the data are not splitted and only one
        # Dataset object is returned. In this case no transformations are applied
        if not percentages:
            datasets.append(Ai4MarsData(Xt, yt))
            return datasets[0]

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

        # Convertion to torch tensors
        for i in range(len(subsets_X)):                             
            subsets_X[i] = torch.stack(subsets_X[i], dim=0).permute(0,3,2,1)
            subsets_y[i] = torch.stack(subsets_y[i], dim=0).permute(0,3,2,1)

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

        # Creation of datasets
        for i in range(len(subsets_X)):

            if transform and i == 1:
                datasets.append(Ai4MarsData(augmented_set_X, augmented_set_y))
            
            else:
                datasets.append(Ai4MarsData(subsets_X[i], subsets_y[i]))

        return datasets
    

class Ai4MarsLightProcessor():
    ...


class Ai4MarsLoad():
    COLAB = 'google.colab' in sys.modules
    LOCAL = not(COLAB)

    if LOCAL:

        from git import Repo

        # Initialize the Git repository object
        repo = Repo(".", search_parent_directories=True)

        # Get the root directory of the Git project
        root_dir = repo.git.rev_parse("--show-toplevel")

        from pathlib import Path

        # Set up path for custom importer modules
        importer_module = root_dir + '/dataloader/'
        sys.path.insert(0, importer_module)

        # Insert here your local path to the dataset
        data_path = '/home/leeoos/Desktop/' #str(input("Path to the dataset: "))
        num_of_images = 200 #int(input("How many images do you want to import: "))
        save_path = '/home/leeoos/Desktop/npdataset/' #str(input("Path to save the npdataset [optional]: "))

        # Import data
        from loader import Ai4MarsImporter, Ai4MarsProcessor, Ai4MarsData

        data_import = Ai4MarsImporter()
        X, y = data_import(path=data_path, num_of_images=num_of_images, save_path=save_path)

        pre_process = False

        if pre_process:
            transform = None
            train_set, test_set, val_set = processor(X, y, percentages=[0.54, 0.26, 0.20], transform=transform)

    elif COLAB:

        from google.colab import drive
        drive.mount('/content/drive')

        # On Colab the path to the module ti fixed once you have
        # corretly set up the project with gitsetup.ipynb
        fixed_path = '/content/drive/MyDrive/Github/visiope/dataloader/'
        sys.path.insert(0, fixed_path)

        # Insert here the path to the dataset on your drive
        data_path = input("Path to the dataset: ")
        num_of_images = int(input("How many images do you want to import: "))

        # Decide is you want the raw dataset or the numpy version
        npdataset = True

        if npdataset:

            # Establish npdataset
            npdataset_X = 'X_' + str(num_of_images)               
            npdataset_y = 'y' + npdataset_X[1:]
            npdataset_path = '/content/dataset/'

            if not(os.path.exists(npdataset_path + npdataset_X) and
                os.path.exists(npdataset_path + npdataset_y)):

                import gdown

                # get url of np dataset (temporarerly my drive)
                if num_of_images == 200:
                    url_X_200 = 'https://drive.google.com/uc?id=1HLAQgjZbGa3lMzdyIvgykCiUJ6P4OaEX'
                    url_y_200 = 'https://drive.google.com/uc?id=1Ue74LEe0WlEnFxRcIl19WyvXR1EyYIsS'

                elif num_of_images == 1000:
                    url_X_1000 = 'https://drive.google.com/uc?id=1zvXKK1qc2PNbAMyq0XWfqrhAWB5kO3yO'
                    url_y_1000 = 'https://drive.google.com/uc?id=1gmnWAGjZJYt-VIpWRK_cEOXQfFfuFAwe'

                # Download np dataset on runtime env
                gdown.download(url_X_1000, data_path, quiet=False)
                gdown.download(url_y_1000, data_path, quiet=False)

        elif not npdataset:

            from loader import Ai4MarsImporter, Ai4MarsProcessor, Ai4MarsData

            data_import = Ai4MarsImporter()
            X, y = data_import(path=data_path, num_of_images=num_of_images)

    else:
        raise Exception('Unknown Environment')
    
if __name__ == '__main__':
    ...