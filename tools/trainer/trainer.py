""" Trainer module for MER-Segmentation 2.0 """

import torch
import time
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt

# This class collects all the training functionalities to train different models
class Ai4MarsTrainer():

    # Initialization of training parameters in the class constructor
    def __init__(self, loss_fn, optimizer, train_loader, val_loader,
                 transform=None, device='cpu', save_state='./'):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_state = save_state
        self.transform = transform
        self.loss_list = []
        self.tloss_list = []
        self.vloss_list = []

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

            # New shape: B x W x H x C
            outputs = outputs.permute(0,2,1,3).permute(0,1,3,2)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)

            # Compute loss gradient
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
                toutputs = toutputs.permute(0,2,1,3).permute(0,1,3,2)
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
        last_loss =  (running_loss) / (batch_index+1)

        # Print report at the end of the last batch
        # and append loss to loss list
        print(f'EPOCH {epoch_index+1}')
        print(f'Train loss: {last_loss}')
        self.loss_list.append(last_loss)
        
        if self.transform:
            last_tloss = running_tloss / (t_index+1)
            print(f'Transformed train loss: {last_tloss}')
            self.tloss_list.append(last_tloss)

        return last_loss

    # This function implements training for multiple epochs
    def train_multiple_epoch(self, model, EPOCHS=100):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        epoch_number = 0 # just an to reference save state epoch
        last_vloss = 0.
        best_vloss = 1_000_000.
        self.loss_list = []
        self.tloss_list = []
        self.vloss_list = []

        for epoch in range(EPOCHS):

            # Make sure gradient tracking is on, and do a pass over the data
            model.train()

            # Start monitoring training time
            start = time.time()

            avg_loss = self.train_one_epoch(model, epoch)

            # End of monitoring time
            end = time.time()

            # We don't need gradients on to do reporting
            model.eval()

            # Test loss
            running_vloss = 0.0
            for vbatch_index, vbatch in enumerate(self.val_loader):

                # Every data instance is a (input, label) pair
                vinputs, vlabels = vbatch

                # Send inputs and labels to GPU
                vinputs = vinputs.to(self.device)
                vlabels = vlabels.to(self.device)

                # Model prediction
                voutputs = model(vinputs)

                # New shape: B x W x H x C
                voutputs = voutputs.permute(0,2,1,3).permute(0,1,3,2)

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
            last_vloss = running_vloss / (vbatch_index + 1)

            print("Time needed for training: " + str(end-start)+ " seconds")

            # Print report at the end of the epoch
            # and append loss to loss list
            print(f'Validation loss: {last_vloss} \n')
            self.vloss_list.append(last_vloss)

            # Track best performance, and save the model's state
            if last_vloss < best_vloss:
                best_vloss = last_vloss
                model_path = self.save_state + 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

    # Plot loss function on train set and validation set after training
    def custom_plot(trainer, model=None, SAVE_PATH:str=None):

        loss_list = np.array(trainer.loss_list)
        vloss_list = np.array(trainer.vloss_list)

        plt.plot(loss_list, label='Training Loss')
        plt.plot(vloss_list, label='Validation Loss')
        plt.title('Training Performances')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.show()

        print(f'Train mean loss: {loss_list.mean()}')
        print(f'Validation mean loss: {vloss_list.mean()}')

        if SAVE_PATH:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save loss on train and validation as np array for future valuations
            np.save(SAVE_PATH + 'loss_' + '{}.npy'.format(timestamp, model.backbone), loss_list)
            np.save(SAVE_PATH + 'vloss_' + '{}.npy'.format(timestamp, model.backbone), vloss_list)

            # Save the plot to a file
            plt.savefig(SAVE_PATH + 'loss_plot_' + '{}.png'.format(timestamp, model.backbone))

    # Plot histogram of model parameters before and after taraining
    def custom_hist(model, SAVE_PATH:str=None):

        # Obtain the parameter values from the trained model
        parameters = []
        for param in model.parameters():
            parameters.extend(param.cpu().flatten().detach().numpy())

        parameters = np.array(parameters)

        # Filter out non-positive values
        parameters = parameters[parameters > 0]

        # Apply logarithmic scaling to the data
        log_data = np.log10(parameters)

        # Set the range and number of bins manually
        bins = np.linspace(min(log_data), max(log_data), 10)

        # Plot the histogram with logarithmic scaling
        plt.hist(log_data, bins=bins, log=True)

        # Set plot title and labels
        plt.xlabel('Logarithmic Scale')
        plt.ylabel('Frequency')
        plt.title('Histogram of Model Parameters')

        # Show the plot
        plt.show()

        if SAVE_PATH:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save the plot to a file
            plt.savefig(SAVE_PATH + 'loss_plot_' + '{}.png'.format(timestamp, model.backbone))


if __name__ == '__main__':
    pass