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

class Trainer():

    def __init__(self, loss_fn, optimizer, training_set, test_set):

        # Initialization of training parameters
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_set = training_set
        self.test_set = test_set

    def __show_inputs__():
        print("Class inputs are: \n \
              1. loss function \n \
              2. optimizer \n \
              3. training set \n"   
        )

    # This function implements training for just one epoch
    def train_one_epoch(self, model):
   
        # To keep track of the last loss when the function is executed 
        # through multiple epochs
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) so that we can track 
        # the batch index and do some intra-epoch reporting.
        for i, data in enumerate(self.training_set):

            # Every data instance is an (input, label) pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()
            
            # Gather data and report every last batch
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.
            
        return last_loss
    
    # This function implements training for multiple epochs
    def train_multiple_epoch(self, model, EPOCHS=100):
        
        # Initialization of report parameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0 # just a counter
        best_tloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = self.train_one_epoch(model)

            # We don't need gradients on to do reporting
            model.train(False)

            # Test loss
            running_tloss = 0.0
            for i, tdata in enumerate(self.test_set):

                # Every data instance is a (test_input, test_label) pair
                test_inputs, test_labels = tdata

                # Model prediction
                toutputs = model(test_inputs)

                # Run test loss
                test_loss = self.loss_fn(toutputs, test_labels)
                running_tloss += test_loss

            avg_tloss = running_tloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_tloss))
            
            # Track best performance, and save the model's state
            if avg_tloss < best_tloss:
                best_tloss = avg_tloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)
            
            epoch_number += 1


if __name__ == '__main__':

    trainer = Trainer()