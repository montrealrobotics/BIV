# How to run the code:
#python main.py --exp_settings="hello,42,True,mse, 5000" --noise_settings="True,uniform,False,False,0.5,0.5,False,3" \\
# --noise_params="1,100,1000,5000,10000,100000,100,2000,30000,400000" --estim_noise_params="100,500,1000,5000"
#########################################################################
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

from utils import str_to_bool, average_noise_mean

# Global varraibles

d_path = d_params.get('d_path')
tr_size = d_params.get('tr_batch_size')
tst_size = d_params.get('test_batch_size')

learning_rate = n_params.get('lr')
epochs = n_params.get('epochs')

# Main 

if __name__ == "__main__":

    # Parse arguments from the commandline
    
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")
    parser.add_argument("--exp_settings", type=str, default="0")
    parser.add_argument("--noise_settings", type=str, default="0")
    parser.add_argument("--noise_params", type=str, default="0")
    parser.add_argument("--estim_noise_params", type=str, default="0")    

    args = parser.parse_args()

    # Define global varriables.

    exp_settings = args.exp_settings.split(",")
    noise_settings = args.noise_settings.split(",")
    noise_params = args.noise_params.split(",")
    estim_noise_params = args.estim_noise_params.split(",")


    # Extract commandline parameters

    tag = exp_settings[0]
    seed = int(exp_settings[1])
    normalize = exp_settings[2]
    loss_type = exp_settings[3]
    model_type = exp_settings[4]
    average_mean_factor = float(exp_settings[5])

    noise = noise_settings[0]
    noise_type = noise_settings[1]
    estimate_noise_params = str_to_bool(noise_settings[2])
    maximum_hetero = str_to_bool(noise_settings[3])
    hetero_scale = float(noise_settings[4])
    coin_fairness = float(noise_settings[5])
    noise_threshold = noise_settings[6]
    threshold_value = float(noise_settings[7])


    noise_params = list(map(lambda x: float(x), noise_params))
    estim_noise_params =  list(map(lambda x: float(x), estim_noise_params))


  
    Get Wandb tags
    tag = [tag,]
    # Initiate wandb client.
    wandb.init(project="IV",tags=tag , entity="khamiesw")
    # Get the api key from the environment variables.
    api_key = os.environ.get('WANDB_API_KEY')
    # login to my wandb account.
    wandb.login(api_key)

    # Define the dataset
    is_noise = str_to_bool(noise)
    noise_type = noise_type

    normz = str_to_bool(normalize)
    noise_thresh = str_to_bool(noise_threshold)
    thresh_value = float(threshold_value)

    # Set expirement seed
    torch.manual_seed(seed)
    # Set experiment id
    exp_id = str(coin_fairness)

    if estimate_noise_params:
        if average_mean_factor>=0 and len(estim_noise_params)//2==2 and coin_fairness<1:
            estim_noise_params[1] = average_noise_mean(average_mean_factor,estim_noise_params[0],coin_fairness)
        dist_data = {"coin_fairness":coin_fairness,"is_params_est":estimate_noise_params, "is_vmax":maximum_hetero, "vmax_scale":hetero_scale ,"data":estim_noise_params}
    else:
        dist_data = {"coin_fairness":coin_fairness,"is_params_est":estimate_noise_params, "data":noise_params}


    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    train_data = UTKface(d_path, transform= trans, train= True, model= model_type, noise=is_noise, noise_type=noise_type, distribution_data = \
                                        dist_data, normalize=normz, noise_threshold = noise_thresh, threshold_value = thresh_value) 
    test_data = UTKface(d_path, transform= trans, train= False, model= model_type, normalize=normz)

    # Load the data
    train_loader = DataLoader(train_data, batch_size=tr_size)
    test_loader = DataLoader(test_data, batch_size=tst_size)

    # Model
    if loss_type == "iv":
        loss = IVLoss(avg_batch=False)
    elif loss_type == "biv":
        loss = IVLoss(avg_batch=True)
    else:
        loss = torch.nn.MSELoss()

    if model_type == "resnet":
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512,1)   # converting resnet to a regression layer
    else:
        model = AgeModel()
    
    optimz = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(experiment_id=exp_id, train_loader= train_loader, test_loader= test_loader, \
        model=model, loss= loss, optimizer= optimz, epochs = epochs)


    #Call wandb to log model performance.
    wandb.watch(model)
    # train the model
    trainer.train(alogrithm=loss_type)

