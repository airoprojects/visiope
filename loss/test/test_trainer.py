# # Check if the script is runned in local or on colab
# IN_COLAB = 'google.colab' in sys.modules

# # Path object for the current file
# current_file = str(Path(__file__).parent.parent)

# # Setup path for custom imports in local and colab
# if IN_COLAB: sys.path.insert(0, '/content/drive/MyDrive/Github/visiope/loss')
# else: sys.path.insert(0, current_file)
# from custom_loss import *

testing_loss = True
if testing_loss:

    from pathlib import Path
    import sys
    if True:
        # Path object for the current file
        current_file = str(Path(__file__).parent.parent)
        sys.path.insert(1, current_file)
    else:
        sys.path.insert(0, '/content/drive/MyDrive/Github/visiope/loss')

    print(sys.path)
    from trainer_module import *
    from custom_loss import *

    loss_fn = ...
    optimizer = ...
    training_set = ...
    test_set = ... 
    trainer = MyTrainer(loss_fn, optimizer, training_set, test_set)