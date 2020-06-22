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
from losses import IVLoss
from train import Trainer

# expirement settings
from params import d_params
from params import n_params

from utils import str_to_bool

# Global varraibles

seed = d_params.get('seed')
d_path = d_params.get('d_path')
tr_size = d_params.get('tr_batch_size')
tst_size = d_params.get('test_batch_size')

learning_rate = n_params.get('lr')
epochs = n_params.get('epochs')

# Main 

if __name__ == "__main__":

    # Parse arguments from the commandline
    
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")

    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--normalize", type=str, default="False")

    parser.add_argument("--noise", type=str,default="False")
    parser.add_argument("--noise_type", type=str,default="uniform")

    # type of the uniform that been added.
    parser.add_argument("--noise_complexity", type=str, default="simple")
    # coin fairness for controlling noises sampling.
    parser.add_argument("--coin_fairness" , type=str, default="fair")

    # mu and v for the first distribution.
    parser.add_argument("--mu" , type=str, default="0")
    parser.add_argument("--v", type= str, default="1")

    # mu and v for the second uniform distribution.
    parser.add_argument("--mu_unf_2" , type=str, default="0")
    parser.add_argument("--v_unf_2", type= str, default="1")

    # is_vmax and vmax_scale for the first distribution.
    parser.add_argument("--uniform_vmax", type=str, default="False")
    parser.add_argument("--vmax_scale", type=str, default="1")

    # is_vmax and vmax_scale for the second distribution.
    parser.add_argument("--uniform_vmax_unf_2", type=str, default="False")
    parser.add_argument("--vmax_scale_unf_2", type=str, default="1")

    parser.add_argument("--loss_type", type=str, default="mse")


    parser.add_argument("--noise_threshold", type=str, default="False")
    parser.add_argument("--threshold_value", type=str, default="1")

    args = parser.parse_args()
    
    # Get Wandb tags
    tag = [args.tag,]
    # Initiate wandb client.
    wandb.init(project="IV",tags=tag , entity="khamiesw")
    # Get the api key from the environment variables.
    api_key = os.environ.get('WANDB_API_KEY')
    # login to my wandb account.
    wandb.login(api_key)
    
    # Set expirement seed
    torch.manual_seed(int(args.seed))
    # Set experiment id
    exp_id = args.mu +"_"+ args.v

    # Define the dataset
    is_noise = str_to_bool(args.noise)
    noise_type =args.noise_type

    normz = bool(args.normalize)
    noise_thresh = str_to_bool(args.noise_threshold)
    thresh_value = float(args.threshold_value)
    loss_type = args.loss_type

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])


    if noise_type == 'uniform':
        dist_data = {"noise_complexity": args.noise_complexity,"coin_fairness":float(args.coin_fairness), "mu": float(args.mu), "v": float(args.v), "is_vmax":str_to_bool(args.uniform_vmax), "vmax_scale":float(args.vmax_scale),\
            "mu_unf_2": float(args.mu_unf_2), "v_unf_2": float(args.v_unf_2), "is_vmax_unf_2":str_to_bool(args.uniform_vmax_unf_2),\
             "vmax_scale_unf_2":float(args.vmax_scale_unf_2) }
    elif noise_type == 'gamma':
        dist_data = {"mu": float(args.mu), "v": float(args.v)}


    train_data = UTKface(d_path, transform= trans, train= True, noise=is_noise, noise_type=noise_type, distribution_data = \
                                        dist_data, normalize=normz, noise_threshold = noise_thresh, threshold_value = thresh_value) 
    test_data = UTKface(d_path, transform= trans, train= False, normalize=normz)

    # Load the data
    train_loader = DataLoader(train_data, batch_size=tr_size)
    test_loader = DataLoader(test_data, batch_size=tst_size)

    train_dataset = train_loader
    test_dataset = test_loader 

    # Model
    if loss_type == "iv":
        loss = IVLoss(avg_batch=False)
    elif loss_type == "biv":
        loss = IVLoss(avg_batch=True)
    else:
        loss = torch.nn.MSELoss()

    model = AgeModel()
    optimz = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(experiment_id=exp_id, train_loader= train_dataset, test_loader= test_dataset, \
        model=model, loss= loss, optimizer= optimz, epochs = epochs)


    # #Call wandb to log model performance.
    wandb.watch(model)
    # # train the model
    trainer.train(alogrithm=loss_type)

