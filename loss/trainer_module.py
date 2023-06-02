""" Trainer module for MER-Segmentation """

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

# Training functions
def trainer(hyperparameters, model, multiple_epochs=False, epoch_index=0, tb_writer=0):

    if multiple_epochs: 
        train_multiple_epoch(hyperparameters, EPOCHS=100)
    else:
        train_one_epoch(parameters, epoch_index, tb_writer)
    
# IMPORTANT: FIND OUT ABOUT TB_WRITER
def train_one_epoch(parameters, epoch_index, tb_writer):

    # Initialization of training parameters
    model = parameters['model']
    loss_fn = parameters['loss']
    optimizer = parameters['optimizer']
    training_set = parameters['training']
    device = parameters['device']

    # To keep track of the last loss when the function is executed 
    # through multiple epochs
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) so that we can track 
    # the batch index and do some intra-epoch reporting.
    for i, data in enumerate(training_set):

        # Every data instance is an (input, label) pair
        inputs, labels = data

        ''' 
        This part should be done by the dataloader on all the dataset
        inputs = inputs.permute(0,3,1,2).to(device)
        labels = labels.to(device)
        '''

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        '''
        This part should be done by the dataloader on all the dataset
        labels = labels [None, :, :, :]
        labels = labels.type(torch.DoubleTensor)
        '''

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        '''
        IMPORTANT: UNCOMMENT THIS PART AFTER FINDING OUT ABOUT TB_WRITER
        '''
        # Gather data and report every last batch
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_set) + i + 1
            '''tb_writer.add_scalar('Loss/train', last_loss, tb_x)'''
            running_loss = 0.
        
    return last_loss


def train_multiple_epoch(parameters, EPOCHS=100):

    # Initialization of training parameters
    model = parameters['model']
    loss_fn = parameters['loss']
    test_set = parameters['validation']
    device = parameters['device']

    # Initialization of report parameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = 0 #SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0 # just a counter
    best_tloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        # Test loss
        running_tloss = 0.0
        for i, tdata in enumerate(test_set):

            # Every data instance is a (test_input, test_label) pair
            tinputs, tlabels = tdata

            '''
            This part should be done by the dataloader on all the dataset
            tlabels = tlabels.type(torch.DoubleTensor)
            tlabels = tlabels.to(device)
            '''

            # Model prediction
            toutputs = model(tinputs)

            # Run test loss
            tloss = loss_fn(toutputs, tlabels)
            running_tloss += tloss

        avg_tloss = running_tloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_tloss))

        '''
        # IMPORTANT: UNCOMMENT THIS PART AFTER FINDING OUT ABOUT TB_WRITER
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()
        '''
        
        # Track best performance, and save the model's state
        if avg_tloss < best_vloss:
            best_vloss = avg_tloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
        
        epoch_number += 1

if __name__ == '__main__':

    # This variable needs to be initialized with dummy values/data to test loss integration

    model = None
    optimizer = None
    loss_fn = None
    training_set = None 
    validation_set = None
    device = None

    parameters = {
        'model' : model,
        'loss' : loss_fn,
        'optimizer' : optimizer,
        'training' : training_set,
        'validation': validation_set,
        'device' : device
    }

    print(IN_COLAB)

