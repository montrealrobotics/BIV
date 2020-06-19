import shutil 
import os
import itertools
from datetime import datetime

import pandas as pd

import torch
from torch.nn import MSELoss

import matplotlib.pyplot as plt

from utils import group_testing, group_labels, get_dataset_stats, normalize_images, normalize_labels, plot_hist
from params import d_params

import wandb



class Trainer:

    """
        Description:
            A class with multiple training methods:
                - Baseline (The model without injecting any noises)
                - Baseline with noise.
                - Baseline with IV loss.
                - Baseline with batch normalized IV loss.

        Args:
            :cuda (bool) [private]: Controls if the model will be run on GPU or not.
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

        if self.cuda:
            print("Running using Cuda support")
            self.model = self.model.cuda(0)

    def save(self, df, path):
        df.to_csv(path)
        wandb.save(path)

    def save_last_epoch(self, lst, path):

        # unpack all tr_out from list of lists to a one list.
        print("********",lst)
        lst_unpck = list(
            itertools.chain.from_iterable(lst))
        print(lst_unpck)
        # 2) Convert the labels and out to dataframes.
        lst_df = pd.DataFrame(lst_unpck, columns=['col'])

        self.save(lst_df, path)

    def zip_results(self, files):
        path = './'  # Current directory
        directory_name = str(self.expermient_id)

        try:
            folder = os.mkdir(path+directory_name)
            for file_name in files:
                shutil.copyfile("./"+ file_name,"./"+path+directory_name+"/"+file_name)
            shutil.make_archive(directory_name,'zip',directory_name)
        except OSError:
            print("zip operation has faild")

    def train(self, alogrithm='default'):
        """
        Description:
            Train the network using IV loss.

        Return:
            Model
        Return type:
            nn.Module

        Args:
            :train_loader: A data loader for the training set.
            :test_loader:  A data loader for the testing set.
            :model: A random model. (i.e CNN).
            :loss: IV (or normalized) loss function.
            :optimizer: An optimizer. (i.e Adam)
            :epochs: Number of epochs
            """

        train_loss_df = pd.DataFrame()
        test_loss_df = pd.DataFrame()

        # Dataframes for saving the training and testing labels into .csv files.
        train_labels_df = pd.DataFrame()
        test_labels_df = pd.DataFrame()

        # Dataframes for saving the training and testing predictions into .csv files.
        train_out_df = pd.DataFrame()
        test_out_df = pd.DataFrame()

        for epoch in range(self.epochs):
            # Saving the train and test losses for logging and visualization purposes.
            tr_losses = []
            tst_losses = []

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
                    if alogrithm == "iv" or alogrithm == "biv":
                        noises_vars = torch.unsqueeze(
                            train_sample[3], 1).type(torch.float32).cuda(0)
                else:
                    tr_batch = train_sample[0]
                    tr_labels = torch.unsqueeze(train_sample[1], 1)
                    if alogrithm == "iv" or alogrithm == "biv":
                        noises_vars = torch.unsqueeze(
                            train_sample[3], 1).type(torch.float32)
                # feeding the data into the model.
                tr_out = self.model(tr_batch)

                # Choose the loss function.
                if alogrithm =="iv" or alogrithm == "biv":
                    mloss = self.loss(tr_out, tr_labels, noises_vars)
                else:
                    mloss = self.mse_loss(tr_out,tr_labels)

                tr_losses.append(mloss.item())

                # Optimize the model.
                mloss.backward()
                self.optimizer.step()
                

                with torch.no_grad():
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
                        tst_losses.append(tloss.item())
                        
                        # log the train and test outputs on the last epoch and the last batch.
                        if epoch == self.last_epoch :#and train_sample_idx == self.train_batches_number-1 :
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


                if epoch == self.last_epoch:
                    time = str(datetime.now())
                    self.save_last_epoch(tr_out_lst_epoch, self.server_path+"train_out.csv")
                    self.save_last_epoch(tr_lbl_lst_epoch,self.server_path+"train_labels.csv")
                    self.save_last_epoch(tst_out_lst_epoch,self.server_path+"test_out.csv")
                    self.save_last_epoch(tst_lbl_lst_epoch,self.server_path+"test_labels.csv")

                    self.save_last_epoch([tr_losses],self.server_path+"train_loss_"+time+".csv")
                    self.save_last_epoch([tst_losses],self.server_path+"test_loss_"+time+".csv")
                    # Zip all the files and upload them to wandb.
                    self.zip_results([self.server_path+"train_loss_"+time+".csv", self.server_path+"test_loss_"+time+".csv", \
                      self.server_path+"train_out.csv", self.server_path+"train_labels.csv", self.server_path+"test_out.csv", \
                        self.server_path+"test_labels.csv"])
                    # log the results to wandb
            for i in range(len(tr_losses)):
                wandb.log({"train loss": tr_losses[i], "test loss": tst_losses[i]})

                print("Batch {} has finished".format(train_sample_idx))
            print('Epoch:{} has finished'.format(epoch))
