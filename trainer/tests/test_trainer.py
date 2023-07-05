import sys


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

    fixed_path_trainer = '/content/drive/MyDrive/Github/visiope/trainer/'
    sys.path.insert(0, fixed_path_loss)

    # Insert here the path to the dataset on your drive
    data_path = '/content/drive/MyDrive/Dataset/'

print(sys.path)

from astrainer import Ai4MarsTrainer

loss_fn = ...
optimizer = ...
training_set = ...
test_set = ... 
trainer = Ai4MarsTrainer(loss_fn, optimizer, training_set, test_set)