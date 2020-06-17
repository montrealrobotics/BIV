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
    parser.add_argument("--mu" , type=str, default="0")
    parser.add_argument("--v", type= str, default="1")
    parser.add_argument("--normalize", type=str, default="False")


    parser.add_argument("--noise", type=str,default="False")
    parser.add_argument("--noise_type", type=str,default="uniform")
    parser.add_argument("--uniform_vmax", type=str, default="False")
    parser.add_argument("--vmax_scale", type=str, default="1")

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

    # Define the dataset
    is_noise = str_to_bool(args.noise)
    noise_type =args.noise_type

    normz = bool(args.normalize)
    noise_thresh = str_to_bool(args.noise_threshold)
    thresh_value = float(args.threshold_value)
    loss_type = args.loss_type

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    if noise_type == 'uniform':
        dist_data = (float(args.mu), float(args.v), str_to_bool(args.uniform_vmax), float(args.vmax_scale) )
    elif noise_type == 'gamma':
        dist_data = (float(args.mu), float(args.v))


    # print('tag',args.tag)
    # print('seed',int(args.seed))
    # print('is noise: ',is_noise)
    # print('n_type',noise_type)
    # print('normalize:',normz)
    # print('noise-thresh',noise_thresh)
    # print('mu',float(args.mu))
    # print('v', float(args.v))
    # print('uniform_vmax', str_to_bool(args.uniform_vmax))
    # print('vmax_scale', float(args.vmax_scale))
    # print('loss_type', loss_type)
    # print('thres', noise_thresh)
    # print('thres_value', thresh_value)





    train_data = UTKface(d_path, transform= trans, train= True, noise=is_noise, noise_type=noise_type, distribution_data = dist_data, normalize=normz, \
                                                                                                    noise_threshold = noise_thresh, threshold_value = thresh_value) 
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
    trainer = Trainer()
    optimz = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # #Call wandb to log model performance.
    # wandb.watch(model)
    # # train the model
    # trainer.train_w_iv(train_dataset,test_dataset,model,loss,optimz,epochs)
