""" Trainer module for MER-Segmentation 2.0 """

import time
import torch
import numpy as np
from datetime import datetime 
import matplotlib.pyplot as plt

# This class collects all the training functionalities to train different models
class Ai4MarsTrainer():

    # Initialization of training parameters in the class constructor
    def __init__(self, loss_fn, optimizer, train_loader, val_loader, lr_scheduler=None,
                 transform=None, device='cpu', info:dict={}, model_name:str='', dump:bool=True):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.transform = transform
        self.lr_scheduler = lr_scheduler
        
        self.loss_list = []
        self.tloss_list = []
        self.vloss_list = []

        self.info = info
        self.model_name = model_name
        self.token = str(datetime.now().strftime('%Y%m%d-%H%M%S'))

        if model_name: self.token =  self.token + '-' + model_name

        # Choose wheter save data or just dump them
        if dump: self.location = './dump/'
        else: self.location = './data/'

        # Keep path to model results as attribute for futire saving
        self.results_path =  self.location + self.token

        SAVE_PATH = self.results_path + '/model_state/'  

        import os
        if not os.path.exists(SAVE_PATH) : 
            os.makedirs(SAVE_PATH)

        if self.info:

            with open(SAVE_PATH + 'config', 'w') as config:
                self.info = [
                    "Dataset: " + self.info['dataset'],
                    "Model name: " + str(self.model_name),
                    "Image size: " + str(self.info['size']),
                    "Splitting percentages: " + self.info['percentages'],
                    "Prior augmentation: " + str(self.info['transform']),
                    "Online augmentation: " + str(self.transform)
                ]
                config.writelines(["%s\n" % item  for item in self.info])


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

            # New shape: B x W x H x C
            outputs = outputs.permute(0,2,1,3).permute(0,1,3,2)

            # Adjust label to be 2D tensors of batch size
            labels = labels.squeeze()
            labels = labels.long()

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)

            # Compute loss gradient
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            running_loss += loss.item()

            # if transformation exists apply them to the batch
            if self.transform:

                # Transform inputs and labes with same transformation
                state = torch.get_rng_state()   # save tensor state
                tinputs = self.transform(inputs)
                torch.set_rng_state(state)      # load tensor state
                tlabels = self.transform(labels)

                self.optimizer.zero_grad()

                tinputs = tinputs.to(self.device)
                tlabels = tlabels.to(self.device)

                toutputs = model(tinputs)
                toutputs = toutputs.permute(0,2,1,3).permute(0,1,3,2)

                labels = labels.squeeze()
                labels = labels.long()

                tloss = self.loss_fn(toutputs, tlabels)
                tloss.backward()
                self.optimizer.step()
                running_tloss = tloss.item()

                t_index += 1
                tinputs.detach()
                tlabels.detach()
                del tinputs
                del tlabels

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
    def train_multiple_epoch(self, model, EPOCHS:int=100, SAVE_PATH:str='./'):
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

                SAVE_PATH = SAVE_PATH = self.results_path + '/model_state/'

                torch.save(model.state_dict(), SAVE_PATH + 'model_{}_{}'.format(timestamp, epoch_number))

            if self.lr_scheduler:   
                self.lr_scheduler.step(val_loss)
            

    # Plot loss function on train set and validation set after training
    def plot_loss(self, SAVE_PATH:str='.'):

        loss_list = np.array(self.loss_list)
        vloss_list = np.array(self.vloss_list)

        plt.plot(loss_list, label='Training Loss')
        plt.plot(vloss_list, label='Validation Loss')

        if self.tloss_list:
            tloss_list = np.array(self.tloss_list)
            plt.plot(tloss_list, label='Transformation Loss')

        plt.title('Training Performances')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        
        if SAVE_PATH:

            import os 

            SAVE_PATH = self.results_path + '/loss/' 

            if not os.path.exists(SAVE_PATH) : 
                os.makedirs(SAVE_PATH)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            print(f"Data will be saved in {SAVE_PATH}")

            # Save loss on train and validation as np array for future valuations
            # np.save(SAVE_PATH + 'loss_' + '{}.npy'.format(timestamp), loss_list)
            # np.save(SAVE_PATH + 'vloss_' + '{}.npy'.format(timestamp), vloss_list)

            if self.tloss_list:
                np.save(SAVE_PATH + 'tloss_' + '{}.npy'.format(timestamp), tloss_list)

            # Save the plot to a file
            plt.savefig(SAVE_PATH + 'loss_plot_' + '{}.png'.format(timestamp))

        plt.show()

        print(f'Train mean loss: {loss_list.mean()}')
        print(f'Validation mean loss: {vloss_list.mean()}')

    # Plot histogram of model parameters before and after taraining
    def param_hist(self, model, SAVE_PATH:str='.', label:str=''):

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

        
        if SAVE_PATH:

            import os 

            SAVE_PATH = self.results_path + '/hist/' 

            if not os.path.exists(SAVE_PATH) : 
                os.makedirs(SAVE_PATH)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            print(f"Data will be saved in {SAVE_PATH}")

            # Save the plot to a file
            plt.savefig(SAVE_PATH + 'parameters_hist_' + label + '_{}.png'.format(timestamp, model.backbone))

        # Show the plot
        plt.show()


if __name__ == '__main__':
    pass