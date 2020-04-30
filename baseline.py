###################################################################################################

# Things to do before ruuning the code on the server:

# Change the dataset directory on the dataloder from: "./Dataset/UTKface/*" to "/datasets/UTKface/*"
# Change Wandb log file from: test      to: iv_w_baseline
###################################################################################################


import argparse
import os

import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import  DataLoader
from torchvision import transforms

import wandb


from dataloader import UTKface
from model import AgeModel
from train import Trainer

# expirement settings
from params import d_params
from params import n_params


# Global varraibles

seed = d_params.get('seed')
d_path = d_params.get('d_path')
tr_size = d_params.get('tr_batch_size')
tst_size = d_params.get('test_batch_size')

learning_rate = n_params.get('lr')
epochs = n_params.get('epochs')


# Configure Settings
torch.manual_seed(seed)

# Wandb settings
wandb.init(project="IV", entity="khamiesw")

# Get the api key from the environment variables.
api_key = os.environ.get('WANDB_API_KEY')
# login to my wandb account.
wandb.login(api_key)



# Main 

if __name__ == "__main__":

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    train_data = UTKface(d_path, transform= trans, train= True, noise=False) 
    test_data = UTKface(d_path, transform= trans, train= False)

    train_loader = DataLoader(train_data, batch_size=tr_size)
    test_loader = DataLoader(test_data, batch_size=tst_size)


# Model
    model = AgeModel()
    loss = torch.nn.MSELoss()
    trainer = Trainer()
    optimz = torch.optim.Adam(model.parameters(), lr=learning_rate)
       

    train_dataset = train_loader
    test_dataset = iter(test_loader).next()


    wandb.watch(model)
    trainer.train(train_dataset,test_dataset,model,loss,optimz,epochs)
