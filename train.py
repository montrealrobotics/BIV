import shutil 
import os
import itertools
import uuid
from datetime import datetime

import pandas as pd
import numpy as np

import torch
from torch.nn import MSELoss

from settings import d_params


class Trainer:

    """
        Description:
            This is the main class that is responsbile for training the models. It achieves that through:

                1) train: 
                    A function that responsible for doing the training and testing operation. It uses mini-batch training setting. 
                2) Zip results:
                    A method that respinsible for zaipping the outputs of the model and the corresponding statistics and upload them to WandB servers.

        Args:
            :expermient_id: An experiment id for distinguishing the result files for each experiment.
            :train dataloader: A dataloader for the training data.
            :test dataloader: A dataloader for the testing data.
            :model: The model that is need to be trained.
            :loss: A loss function to measure the model's performance.
            :optimizer: An optimizer to optimize model parameters in the light of the loss function.
            :epochs: Number of training epochs.
    """

    def __init__(self, experiment_id, train_loader, test_loader, model, loss, optimizer, epochs):

        self.expermient_id = experiment_id
        self.cuda = torch.cuda.is_available()  # Check Cuda avaliability
        self.train_data = train_loader
        self.test_data = test_loader
        self.train_batches_number = len(train_loader)
        self.test_batches_number = len(test_loader)
        self.model = model
        self.loss = loss
        self.mse_loss = MSELoss()
        self.optimizer = optimizer
        self.epochs = epochs
        self.last_epoch = self.epochs-1
        self.server_path = d_params.get('server_path')
        self.uniqueID = str(uuid.uuid4().fields[-1])[:6]

        self.save_path = os.path.join(self.server_path, experiment_id+"_"+self.uniqueID)
        os.mkdir(self.save_path)



        if self.cuda:
            print("Running using Cuda support")
            self.model = self.model.cuda(0)

    def save(self, df, path):
        df.to_csv(path)

    def save_last_epoch(self, lst, path):

        # unpack all tr_out from list of lists to a one list.
        lst_unpck = list(
            itertools.chain.from_iterable(lst))
        # 2) Convert the labels and out to dataframes.
        lst_df = pd.DataFrame(lst_unpck, columns=['col'])

        self.save(lst_df, path)

    def zip_results(self, files):
        """
        Description:
            A method to zip the results and upload them to WandB server.
        Return:
            0 if success, otherwise -1.
        Return type:
            int

        Args:
            :files: A list of training and testing results (predictions and losses).
            """
        directory_name = str(self.expermient_id)    
        try:
            folder = os.mkdir(os.path.join(self.save_path,directory_name))
            for file_name in files:
                shutil.copyfile(os.path.join(self.save_path,file_name), os.path.join(self.save_path,directory_name,file_name))
            shutil.make_archive(os.path.join(self.save_path,directory_name),'zip', os.path.join(self.save_path,directory_name))
        except OSError:
            print("zip operation has faild")

    def train(self, loss_type='default'):
        """
        Description:
            A method to train the models that are included in this baseline. it has three training settings:

                1) Baseline: Train the model with the non-noisy labels using MSE loss.
                2) Cutoff: Train the model with noisy labels that are filtered using CutoffMSE loss.
                3) BIV: Train the model with noisy labels using BIV loss.

        Return:
            Trained model.
        Return type:
            nn.Module object.

        Args:
            :loss type: Type of the loss function that is used to train the model.
            """
        train_loss_df = pd.DataFrame()
        test_loss_df = pd.DataFrame()

        # Dataframes for saving the training and testing labels into .csv files.
        train_labels_df = pd.DataFrame()
        test_labels_df = pd.DataFrame()

        # Dataframes for saving the training and testing predictions into .csv files.
        train_out_df = pd.DataFrame()
        test_out_df = pd.DataFrame()

        # Saving the train and test losses for logging and visualization purposes.
        tr_losses = []
        tst_losses = []
        for epoch in range(self.epochs):
            # Saving the train and test predictions for logging and visualization purposes.
            tr_out_lst_epoch = []
            tst_out_lst_epoch = []

            # Saving the train and test labels for logging and visualization purposes.
            tr_lbl_lst_epoch = []
            tst_lbl_lst_epoch = []
        
            for train_sample_idx, train_sample in enumerate(self.train_data):
                self.optimizer.zero_grad()
                # Moving data to cuda
                if self.cuda:
                    tr_batch = train_sample[0].cuda(0)
                    tr_labels = torch.unsqueeze(train_sample[1], 1).cuda(0)
                    if loss_type != "mse":
                        noises_vars = train_sample[3].type(torch.float32).cuda(0)
                     
                else:
                    tr_batch = train_sample[0]
                    tr_labels = torch.unsqueeze(train_sample[1], 1)
                    if loss_type != "mse":
                        noises_vars = train_sample[3].type(torch.float32)
                        
                
                
                # feeding the data into the model.
                tr_out = self.model(tr_batch)

                # Choose the loss function.
                if loss_type != "mse":
                    mloss = self.loss(tr_out, tr_labels, noises_vars)
                else:
                    mloss = self.mse_loss(tr_out,tr_labels)
               
                if mloss !="cutoffMSE:NO_LOSS":
                    tr_losses.append(mloss.item())
                    # Optimize the model.
                    mloss.backward()
                    self.optimizer.step()
                
                    with torch.no_grad():
                        tst_b_losses= []
                        for test_sample_idx, test_sample in enumerate(self.test_data):
                            if self.cuda:
                                tst_batch = test_sample[0].cuda(0)
                                tst_labels = torch.unsqueeze(test_sample[1], 1).cuda(0)
                            else:
                                tst_batch = test_sample[0]
                                tst_labels = torch.unsqueeze(test_sample[1], 1)

                            # feed the data into the model.

                            tst_out = self.model(tst_batch)
                            # estimate the loss.
                            tloss = self.mse_loss(tst_out, tst_labels)
                            # append the loss.
                            tst_b_losses.append(tloss.item())
                        
                            # log the train and test outputs on the last epoch and the last batch.
                            if epoch == self.last_epoch and train_sample_idx == self.train_batches_number-1 :
                                # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                                tr_out_lst_epoch.append(
                                    tr_out.view(1, -1).squeeze(0).tolist())
                                tr_lbl_lst_epoch.append(
                                tr_labels.view(1, -1).squeeze(0).tolist())

                                # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                                tst_out_lst_epoch.append(
                                    tst_out.view(1, -1).squeeze(0).tolist())
                                tst_lbl_lst_epoch.append(
                                tst_labels.view(1, -1).squeeze(0).tolist())

                        # Estimating the mean of the losses over the test batches.
                        mean_tst_b_losses = np.mean(tst_b_losses)
                        tst_losses.append(mean_tst_b_losses)    

                if epoch == self.last_epoch and train_sample_idx == self.train_batches_number-1:
                    self.save_last_epoch(tr_out_lst_epoch, os.path.join(self.save_path,"train_outs.csv"))
                    self.save_last_epoch(tr_lbl_lst_epoch, os.path.join(self.save_path,"train_labels.csv"))
                    self.save_last_epoch(tst_out_lst_epoch,os.path.join(self.save_path,"test_outs.csv"))
                    self.save_last_epoch(tst_lbl_lst_epoch,os.path.join(self.save_path,"test_labels.csv"))
                
                #print("******************** Batch {} has finished ********************".format(train_sample_idx))
            print('#################### Epoch:{} has finished ####################'.format(epoch))

        
        self.save_last_epoch([tr_losses],os.path.join(self.save_path,"train_losses.csv"))
        self.save_last_epoch([tst_losses],os.path.join(self.save_path,"test_losses.csv"))
        self.zip_results(["train_losses.csv", "test_losses.csv", \
            "train_outs.csv", "train_labels.csv", "test_outs.csv", "test_labels.csv"])

        return self.model