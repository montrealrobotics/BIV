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


# Configure global settings

torch.manual_seed(42)
wandb.init(project="iv_w_baseline", entity="khamiesw")

# Get the api key from the environment variables.
api_key = os.environ.get('WANDB_API_KEY')
# login to my wandb account.
wandb.login(api_key)



# Main 

if __name__ == "__main__":

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    train_data = UTKface("/datasets/UTKFace/*", transform= trans, train= True) 
    test_data = UTKface("/datasets/UTKFace/*", transform= trans, train= False)

    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=1000)

    epochs = 20
    model = AgeModel()
    loss = torch.nn.MSELoss()
    trainer = Trainer()
    optimz = torch.optim.Adam(model.parameters(), lr=3e-4)
       

    train_dataset = train_loader
    test_dataset = iter(test_loader).next()


    # wandb.watch(model)
    trainer.train(train_dataset,test_dataset,model,loss,optimz,epochs)
