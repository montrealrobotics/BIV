import os
import argparse

import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import  DataLoader
from torchvision import transforms

import wandb

from Dataloaders.utkf_dataloader import UTKface
from Dataloaders.wine_dataloader import WineQuality
from model import AgeModel, WineModel
from losses import IVLoss
from train import Trainer

# Import default expirement settings

from params import d_params
from params import n_params

# Import helper tools
from utils import print_experiment_information, assert_arguments, str_to_bool, average_noise_mean


# Main 

if __name__ == "__main__":

    # Parse arguments from the commandline
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")
    parser.add_argument("--experiment_settings", type=str, default="0")
    parser.add_argument("--model_settings", type=str,default="0")
    parser.add_argument("--noise", type=str, default="False") 
    parser.add_argument("--noise_settings", type=str, default="0")
    parser.add_argument("--average_variance", type=str, default="2000") 
    parser.add_argument("--params_settings", type=str, default="0")
    parser.add_argument("--parameters", type=str, default="0")
    parser.add_argument("--epsilon", type=str, default="0.5")
    parser.add_argument("--distributions_ratio", type=str, default="0")

    # Extract commandline parameters   
    args = parser.parse_args()
    experiment_settings = args.experiment_settings.split(",")
    model_settings = args.model_settings.split(",")
    noise_settings = args.noise_settings.split(",")
    params_settings = args.params_settings.split(",")
    parameters = args.parameters.split(",")


    tag = experiment_settings[0]
    seed = experiment_settings[1]
    dataset = experiment_settings[2]
    normalize = experiment_settings[3]

    model_type = model_settings[0]
    loss_type = model_settings[1] 
    epsilon = model_settings[2] 
    
    noise = noise_settings[0] 
    noise_type = noise_settings[1]
    threshold_value = noise_settings[2]

    params_type = params_settings[0]
    hetero_scale = params_settings[1]
    distributions_ratio = params_settings[2] 
    average_variance = params_settings[3] 

    is_estim_noise_params = True if params_type == "meanvar" or params_type == "alphabeta" else False
    


    arguments = {"tag": tag, "seed": seed, "dataset": dataset, "normalize": normalize, "loss_type": loss_type, "model_type": model_type, "epsilon": epsilon,
                 "noise": noise, "average_variance": average_variance, "noise_type": noise_type, "is_estim_noise_params": is_estim_noise_params, 'params_type':params_type,
                 'parameters':parameters,"hetero_scale": hetero_scale, "distributions_ratio": distributions_ratio, "threshold_value": threshold_value}
    
    # Print experiments information
    print_experiment_information(arguments)

     # Assert CommandLine arguments's values
    assert_arguments(arguments)
    
    # Convert commandline arguments to appropriate datatype.
    seed = int(seed)
    normalize = str_to_bool(normalize)
    epsilon = float(epsilon)

    average_variance = float(average_variance)
    is_noise = str_to_bool(noise)
    hetero_scale = float(hetero_scale)
    maximum_hetero = True if hetero_scale>=0 else False 
    distributions_ratio = float(distributions_ratio)
    threshold_value = float(threshold_value)
    is_noise_threshold = True if threshold_value>=0 else False
    
    parameters = list(map(lambda x: float(x), parameters))
  

    # Get Wandb tags
    tag = [tag,]
   # Initiate wandb client.
    wandb.init(project="iv_deep_learning",tags=tag , entity="montreal_robotics")
    # Get the api key from the environment variables.
    api_key = os.environ.get('WANDB_API_KEY')
    # login to my wandb account.
    wandb.login(api_key)

    # Set expirement seed
    torch.manual_seed(seed)
    # Set experiment id
    exp_id = str(distributions_ratio)

    # Define the dataset
    if is_estim_noise_params:
        if average_variance>=0 and len(parameters)//2==2 and distributions_ratio<1:
            parameters[1] = average_noise_mean(noise_type,average_variance,parameters[0],parameters[3],distributions_ratio)
        dist_data = {"coin_fairness":distributions_ratio,"is_params_est":is_estim_noise_params, "is_vmax":maximum_hetero, "vmax_scale":hetero_scale ,"data":parameters}

    else:
            dist_data = {"coin_fairness":distributions_ratio,"is_params_est":is_estim_noise_params, "data":parameters}


    if dataset == "utkf":

        d_path = d_params.get('d_path')
        tr_size = d_params.get('tr_batch_size')
        tst_size = d_params.get('test_batch_size')
        learning_rate = n_params.get('lr')
        epochs = n_params.get('epochs')

        trans= torchvision.transforms.Compose([transforms.ToTensor()])

        train_data = UTKface(d_path, transform= trans, train= True, model= model_type, noise=is_noise, noise_type=noise_type, distribution_data = \
                                            dist_data, normalize=normalize, noise_threshold = is_noise_threshold, threshold_value = threshold_value) 
        test_data = UTKface(d_path, transform= trans, train= False, model= model_type, normalize=normalize)


    elif dataset == "wine":

        d_path = d_params.get('wine_path')
        tr_size = d_params.get('wine_tr_batch_size')
        tst_size = d_params.get('wine_test_batch_size')
        learning_rate = n_params.get('wine_lr')
        epochs = n_params.get('epochs')

        train_data = WineQuality(d_path, train= True, model= model_type, noise=is_noise, noise_type=noise_type, distribution_data = \
                                        dist_data, normalize=normalize, noise_threshold = is_noise_threshold, threshold_value = threshold_value) 
        test_data = WineQuality(d_path, train= False, model= model_type, normalize=normalize)

        
    # Load the data
    train_loader = DataLoader(train_data, batch_size=tr_size)
    test_loader = DataLoader(test_data, batch_size=tst_size)

    if model_type =="vanilla_ann" and dataset=='wine':
        model = WineModel()
        print("#"*80,"Model is:{}".format(model_type), "#"*80)
    elif model_type == "resnet" and dataset == 'utkf':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512,1)   # converting resnet to a regression layer
        print("#"*80,"Model is:{}".format(model_type), "#"*80)
    elif model_type == "vanilla_cnn" and dataset == 'utkf':
        model = AgeModel()
        print("#"*80," Model is:{}".format(model_type, "#"*80))
    else:
        raise ValueError(" Model is not recognized or the dataset and the model are not compatible with each other.")


    # Optimizer
    optimz = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Loss function
    if loss_type == "iv":
        loss = IVLoss(epsilon=epsilon, avg_batch=False)
    elif loss_type == "biv":
        loss = IVLoss(epsilon=epsilon, avg_batch=True)
    else:
        loss = torch.nn.MSELoss()


    # Trainer
    trainer = Trainer(experiment_id=exp_id, train_loader= train_loader, test_loader= test_loader, \
        model=model, loss= loss, optimizer= optimz, epochs = epochs)


    # Call wandb to log model performance.
    wandb.watch(model)
    # train the model
    trainer.train(alogrithm=loss_type)


