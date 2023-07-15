# Data Utils

Documentation for the classes inside [utils.py](https://github.com/airoprojects/visiope/blob/main/tools/data/utils.py):
---
1. *Ai4MarsDataset*: define a custom dataset for the specifick semantic segmentation tast.

    - Inizialization: Ai4MarsDataset(X, y, transform):

        - **X**: torch tensor of inputs images [3 x H x W]
        - **y**: torch tensor of given labes (must match X) [1 x H x W]
        - **transform**: possible sequence of torch transformation to apply to each item of the class while retriving it

---
2. *Ai4MarsDownload*: allow to download the dataset both in colab or on a local machine. It will return the data as a pair X, y where they are both torch tensors. The former rapresent the input images while the latter represent the labels.

    - Inizialization Ai4MarsDownload()

    - Call: downloader = Ai4MarsDownload() --> downloader(self, PATH:str='./', NUM_IMAGES:int=200, SAVE_PATH:str=None): 


