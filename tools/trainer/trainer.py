""" Trainer module for MER-Segmentation 2.0 """

import torch
import time
from datetime import datetime 

# This class collects all the training functionalities to train different models
class Ai4MarsTrainer():

    # Initialization of training parameters in the class constructor
    def __init__(self, loss_fn, optimizer, train_loader, val_loader,
                 transform=None, device='cpu', save_state=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_state = save_state
        self.transform = transform
        self.loss_list = []
        self.tloss_list = []

    # This function implements training for just one epoch
    def train_one_epoch(self, model, epoch_index=0):
        running_loss = 0.
        last_loss = 0.

        # parameters for online data augmentation (batch transformation)
        running_tloss = 0.
        last_tloss = 0.
        t_index = 0 

        for batch_index, batch in enumerate(self.train_loader):
            # Every data instance is an (input, label) pair
            inputs, labels = batch

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Send inputs and labels to GPU (or whatever device is)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Make predictions for this batch
            outputs = model(inputs)

            # Adjust label to be 2D tensors of batch size
            labels = labels.squeeze()
            labels = labels.long()

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            running_loss += loss.item()

            # if transformation exists apply them to the batch
            if self.transform:
                tinputs = self.transform(inputs)
                tinputs = tinputs.to(self.device)
                self.optimizer.zero_grad()
                toutputs = model(tinputs)
                tloss = self.loss_fn(toutputs, labels)
                tloss.backward()
                running_tloss = tloss.item()
                t_index += 1
                tinputs.detach()
                del tinputs

            # Free up RAM/VRAM
            inputs.detach()
            labels.detach()
            del inputs
            del labels

        # Compute the average loss over all batches
        last_loss =  running_loss / (batch_index + 1)

        # Print report at the end of the last batch
        print(f'Epoch {epoch_index+1}')
        print(f'LOSS ON TRAIN: {last_loss}')
        
        if self.transform:
            last_tloss = running_tloss / (t_index + 1)
            print(f'LOSS ON TRANSFORMED-TRAIN: {last_tloss}')

        else:
            last_tloss = None

        return last_loss, last_tloss

    # This function implements training for multiple epochs
    def train_multiple_epoch(self, model, EPOCHS=100):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0 # just a counter
        best_vloss = 1_000_000.
        self.loss_list = []
        self.tloss_list = []

        for epoch in range(EPOCHS):
            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)

            # Start monitoring training time
            start = time.time()

            avg_loss = self.train_one_epoch(model, epoch)

            end = time.time()

            # We don't need gradients on to do reporting
            model.train(False)

            # Test loss
            running_vloss = 0.0
            for vbatch_index, vbatch in enumerate(self.test_loader):

                # Every data instance is a (input, label) pair
                vinputs, vlabels = vbatch

                # Send inputs and labels to GPU
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                # Model prediction
                voutputs = model(vinputs)

                # Send inputs and labels to GPU (or whatever device is)
                vlabels = vlabels.squeeze()
                vlabels = vlabels.long()

                # Run validation loss
                val_loss = self.loss_fn(voutputs, vlabels)
                running_vloss += val_loss.item()

                # Free up RAM/VRAM
                vinputs.detach()
                vlabels.detach()
                del vinputs
                del vlabels
                torch.cuda.empty_cache()

            # Compute the average loss over all batches
            avg_vloss = running_vloss / (vbatch_index + 1)

            print("Time needed for training: " + str(end-start)+ " seconds")

            # Print report at the end of the epoch
            print(f'LOSS ON VALIDATION: {avg_vloss}')

            # Save loss in a list to then perform metrics evaluation
            self.loss_list.append((avg_loss[0], end-start))
            
            # If online data augmentation has been performed:
            if avg_loss[1]:
                self.tloss_list.append((avg_loss[1], end-start))

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = self.save_state + 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    pass
