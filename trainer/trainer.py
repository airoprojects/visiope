""" Trainer module for MER-Segmentation 2.0 """

import sys
import torch
from pathlib import Path
from datetime import datetime

# This class collects all the training functionalities to train different models
class Ai4MarsTrainer():

    # Initialization of training parameters in the class constructor
    def __init__(self, loss_fn, optimizer, train_loader, test_loader):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    # This function implements training for just one epoch
    def train_one_epoch(self, model, epoch_index=0):
        accumulated_loss = 0.
        last_loss = 0.

        for batch_index, batch in enumerate(self.train_loader):
            # Every data instance is an (input, label) pair
            inputs, labels = batch

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            
            # 
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Make predictions for this batch
            outputs = model(inputs)
            
            # TEMPORARY MODIFICATIONS
            #NEED TO MODIFY FOR EACH MODEL, BASED ON LABEL!!!!!!!!!!!!
            new_pred = torch.argmax(outputs, dim=1)
            #new_pred = new_pred[None, :, :, :] FOR MODEL -> UNKNOWN 
            new_pred = new_pred[:, None, :, :] 
            new_pred = new_pred.type(torch.float32)
            # END OF TEMPORARY MODIFICATIONS

            # Compute the loss and its gradients
            loss = self.loss_fn(new_pred, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()
                        
            accumulated_loss += loss.item()
            #report_index = len(self.train_loader) -1 

            # free VRAM
            image.detach()
            label.detach()
            del label
            del image 
           
        # Compute the average loss over all batches
        last_loss =  accumulated_loss / batch_index+1 

        # Print report at the end of the last batch
        print(f'Epoch {epoch_index+1} loss: {last_loss}')
            
        return last_loss
    
    # This function implements training for multiple epochs
    def train_multiple_epoch(self, model, EPOCHS=100):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0 # just a counter
        best_tloss = 1_000_000.

        for epoch in range(EPOCHS):
            print(f'EPOCH: {epoch+1}')

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            avg_loss = self.train_one_epoch(model, epoch)

            # We don't need gradients on to do reporting
            model.train(False)

            # Test loss
            accumulated_tloss = 0.0
            for tbatch_index, tbatch in enumerate(self.test_loader):

                # Every data instance is a (input, label) pair
                tinputs, tlabels = tbatch

                # Model prediction
                toutputs = model(tinputs)

                # Run test loss
                test_loss = self.loss_fn(toutputs, tlabels)
                accumulated_tloss += test_loss.item()

            # Compute the average loss over all batches
            avg_tloss = accumulated_tloss / (tbatch_index + 1)

            # Print report at the end of the epoch
            print(f'LOSS train {avg_loss} test {avg_tloss}')
            
            # Track best performance, and save the model's state
            if avg_tloss < best_tloss:
                best_tloss = avg_tloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    pass
