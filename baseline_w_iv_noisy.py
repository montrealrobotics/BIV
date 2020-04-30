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


# Set expirement seed

torch.manual_seed(seed)

# Main 

if __name__ == "__main__":

    # Parse arguments from the commandline
    
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")

    parser.add_argument("--mu" , type=int, default=0)
    parser.add_argument("--v", type= int, default=1)
    parser.add_argument("--unf_vmax_scale", type=str, default="False")
    parser.add_argument("--scale_value", type=int, default=1)
    parser.add_argument("--tag", type=str, default="default")

    args = parser.parse_args()
    
    # Get Wandb tags

    tag = [args.tag,]

    # Initiate wandb client.
    wandb.init(project="IV",tags=tag , entity="khamiesw")
    # Get the api key from the environment variables.
    api_key = os.environ.get('WANDB_API_KEY')
    # login to my wandb account.
    wandb.login(api_key)
    

    unif_data = (args.mu,args.v, args.unf_vmax_scale, args.scale_value) 

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])


    train_data = UTKface(d_path, transform= trans, train= True, noise=True, noise_type='uniform', uniform_data = unif_data) 
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
    trainer.train_w_iv(train_dataset,test_dataset,model,loss,optimz,epochs)
