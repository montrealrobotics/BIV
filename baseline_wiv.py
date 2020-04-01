###################################################################################################

# Things to do before ruuning the code on the server:

# Change the dataset directory on the dataloder from: "./Dataset/UTKface/*" to "/datasets/UTKface/*"
# Change Wandb log file from: test      to: iv_w_noisy_baseline
###################################################################################################

import os
import argparse

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
wandb.init(project="test", entity="khamiesw")

# Get the api key from the environment variables.
api_key = os.environ.get('WANDB_API_KEY')
print("API Key:",api_key)
# login to my wandb account.
wandb.login(api_key)


# Main 

if __name__ == "__main__":

    # Parse arguments from the commandline
    
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")

    parser.add_argument("--mu" , type=int, default=0)
    parser.add_argument("--v", type= int, default=1)

    args = parser.parse_args()
    unif_data = (args.mu,args.v) 

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    # uniform_data = [(2,1),(5,1),(5,4),\
    #                     (10,1),(10,4),(10,10),\
    #                     (20,1),(20,4),(20,10),(20,50),\
    #                     (50,1),(50,4),(50,10),(50,50),(50,150),\
    #                     (150,1),(150,4),(150,10),(150,50),(150,150),(150,1000)]

                             # Take the first sample.

    train_data = UTKface("./Datasets/UTKFace/*", transform= trans, train= True, noise= True, noise_type='uniform', uniform_data = unif_data) 
    test_data = UTKface("./Datasets/UTKFace/*", transform= trans, train= False, noise= False)

    
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
    trainer.train_w_iv(train_dataset,test_dataset,model,loss,optimz,epochs)
