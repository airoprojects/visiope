# Import Data and Modules 

**Cell One**
```
# Custom Imports
COLAB = 'google.colab' in sys.modules
LOCAL = not COLAB

if COLAB:

    # Clone visiope repo on runtime env
    !git clone https://github.com/airoprojects/visiope.git /

    # Install pytorchmetrics
    !pip install torchmetrics

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
from data.utils import Ai4MarsDownload, Ai4MarsSplitter, Ai4MarsDataLoader

# Import Loss
from loss.loss import Ai4MarsCrossEntropy, Ai4MarsDiceLoss

# Import Trainer
from trainer.trainer import Ai4MarsTrainer

# Import Tester
from tester.tester import Ai4MarsTester

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Cell Three**
```
# Dataloader

# Set this to True if you wnat to load directly the dataloader 
# this can be done only on colab and it is useful to avoid runtime crash
LOAD = True

if LOAD and not LOCAL:

    if not(os.path.exists('/content/dataset/')):

        import gdown

        # get url of torch dataset 
        url = 'https://drive.google.com/drive/folders/104YvO3LcU76euuVe-_62eS_Rld-tOZeh?usp=drive_link'

        !gdown --folder {url} -O /content/

    dataset = torch.load("/content/dataset/dataset.pt")
    train_set = dataset[0]
    test_set = dataset[1]
    val_set = dataset[2]

    # Build Ai4MarsDataloader
    loader = Ai4MarsDataLoader()
    train_loader, test_loader, val_loader = loader(
        [train_set, test_set, val_set], [32, 16, 16])

        
elif LOCAL or not LOAD:

    # Insert here your local path to the dataset (temporary on airodrive)
    raise Exception('Remove this line and inset the path to the dataset below')
    data_path = ...

    # Local path to save the dataset
    save_path = root_dir + '/dataset/'

    # Import data as Ai4MarsDataset
    importer = Ai4MarsDownload()
    X, y = importer(PATH=data_path, NUM_IMAGES=500)

    # Split the dataset
    splitter = Ai4MarsSplitter()
    train_set, test_set, val_set = splitter(X, y, [0.7, 0.2, 0.1], SAVE_PATH=save_path, SIZE=128)

    # Build Ai4MarsDataloader
    loader = Ai4MarsDataLoader()
    train_loader, test_loader, val_loader = loader([train_set, test_set, val_set], [32, 16, 16], SIZE=128, SAVE_PATH=save_path)
```
