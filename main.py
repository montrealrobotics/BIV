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
from utils import print_experiment_information, str_to_bool, average_noise_mean


# Main 

if __name__ == "__main__":

    # Global Variables
    distributions_ratio = 1
    threshold_value = -1
    maximum_hetero = False
    hetero_scale = 1
    is_noise_threshold = False
    warning_messages = {"bool":":argument is not boolean.", "datatype":"datatype is not supported.", "value":"argument value is not recognized."}



    # Parse arguments from the commandline
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")
    parser.add_argument("--experiment_settings", type=str, default="0")
    parser.add_argument("--model_settings", type=str,default="0")
    parser.add_argument("--noise_settings", type=str, default="0") 
    parser.add_argument("--params_settings", type=str, default="0")
    parser.add_argument("--parameters", type=str, default="0")

    # Extract commandline arguments   
    args = parser.parse_args()
    experiment_settings = args.experiment_settings.split(",")
    model_settings = args.model_settings.split(",")
    noise_settings = args.noise_settings.split(",")
    params_settings = args.params_settings.split(",")
    parameters = args.parameters.split(",")

    # Access "experiment_settings" arguments
    tag = experiment_settings[0]
    seed = experiment_settings[1]
    assert isinstance(float(seed), float), "Argument: seed: " + warning_messages.get('datatype')
    seed = float(seed)
    dataset = experiment_settings[2]
    assert dataset in ["utkf","wine"], "Argument: dataset: " + warning_messages.get('value')
    normalize = experiment_settings[3]
    assert isinstance( str_to_bool(normalize), bool), "Argument: normalize: " + warning_messages.get('bool')
    normalize = str_to_bool(normalize)


########################################################################################################################################################################
    # Access "model_settings" arguments
    model_type = model_settings[0]
    assert model_type in ["vanilla_ann","vanilla_cnn", "resnet"], "Argument: model_type: " + warning_messages.get('value')
    loss_type = model_settings[1]
    assert loss_type in ["mse", "iv", "biv"], "Argument: loss_type: " + warning_messages.get('value') 

    if len(model_settings) > 2:
        epsilon = model_settings[2]
        assert epsilon.replace('.','',1).isdigit() , "Argument: epsilon: " + warning_messages.get('datatype')
        epsilon = float(epsilon)    

    noise = noise_settings[0] 
    assert isinstance( str_to_bool(noise), bool), "Argument: noise: " + warning_messages.get('bool')
    noise = str_to_bool(noise)
    if noise:
        noise_type = noise_settings[1]
        assert noise_type in ["binary_uniform","uniform","gamma"], "Argument: noise_type: " + warning_messages.get('value')

    if noise and len(noise_settings) >2:
        threshold_value = noise_settings[2]
        assert threshold_value.replace('.','',1).replace('-','',1).isdigit(), "Argument: threshold_value: " +  warning_messages.get('datatype')
        threshold_value = float(threshold_value)
        is_noise_threshold = True 


    if noise:
        params_type = params_settings[0]
        assert params_type in ["meanvar","meanvar_avg","boundaries","alphabeta"], "Argument: params_type: " + warning_messages.get('value')

        if noise_type == "binary_uniform":
            distributions_ratio = params_settings[1] 
            assert float(distributions_ratio)>=0 and float(distributions_ratio)<=1 , "Argument: distributions_ratio: "+ warning_messages.get('value')
            distributions_ratio = float(distributions_ratio)

            if params_type == "meanvar_avg":
                average_variance = params_settings[2] 
                assert average_variance.replace('.','',1).replace('-','',1).isdigit(), "Argument: average_variance: "+ warning_messages.get('value')
                average_variance = float(average_variance)
            
        is_estim_noise_params = False if params_type=="boundaries" or params_type=="alphabeta" else True 

        if noise and params_type=="meanvar" or params_type=="meanvar_avg":
            maximum_hetero = parameters[0]
            assert isinstance( str_to_bool(maximum_hetero), bool), "Argument: maximum_hetero: " + warning_messages.get('bool')
            maximum_hetero = str_to_bool(maximum_hetero)
            if maximum_hetero:
                hetero_scale = parameters[3]
                assert float(hetero_scale)>=0 and float(hetero_scale)<=1 , "Argument: hetero_scale: "+ "argument value is not recognized."
                hetero_scale = float(hetero_scale)
            parameters = parameters[1:]
        
        for item in parameters: assert item.replace('.','',1).replace('-','',1).isdigit() , "Argument: parameters: " + "datatype is not supported."
        parameters = list(map(lambda x: float(x), parameters))
    else:
        noise_type = None
        is_estim_noise_params = False
        params_type = None
        parameters = None

    
   # Print experiments information
    arguments = {"tag": tag, "seed": seed, "dataset": dataset, "normalize": normalize, "loss_type": loss_type, "model_type": model_type, 
                 "noise": noise, "noise_type": noise_type, "is_estim_noise_params": is_estim_noise_params, 'params_type':params_type,
                 'parameters':parameters}
    
    print_experiment_information(arguments)

###########################################################################################################################################################
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
    exp_id = params_type

    # Define the dataset

    if noise and noise_type =="binary_uniform" and params_type=="meanvar_avg":
        parameters[1] = average_noise_mean(noise_type,average_variance,parameters[0],parameters[3],distributions_ratio)
        dist_data = {"coin_fairness":distributions_ratio,"is_params_est":is_estim_noise_params, "is_vmax":maximum_hetero, "vmax_scale":hetero_scale ,"data":parameters}

    elif noise and noise_type == "binary_uniform":
        dist_data = {"coin_fairness":distributions_ratio, "is_params_est":is_estim_noise_params,"is_vmax":maximum_hetero, "vmax_scale":hetero_scale, "data":parameters}
    else:
        dist_data = {"coin_fairness":distributions_ratio, "is_params_est":is_estim_noise_params,"is_vmax":maximum_hetero, "vmax_scale":hetero_scale,"data":parameters}


    if dataset == "utkf":

        d_path = d_params.get('d_path')
        tr_size = d_params.get('tr_batch_size')
        tst_size = d_params.get('test_batch_size')
        learning_rate = n_params.get('lr')
        epochs = n_params.get('epochs')

        trans= torchvision.transforms.Compose([transforms.ToTensor()])

        train_data = UTKface(d_path, transform= trans, train= True, model= model_type, noise=noise, noise_type=noise_type, distribution_data = \
                                            dist_data, normalize=normalize, noise_threshold = is_noise_threshold, threshold_value = threshold_value) 
        test_data = UTKface(d_path, transform= trans, train= False, model= model_type, normalize=normalize)


    elif dataset == "wine":

        d_path = d_params.get('wine_path')
        tr_size = d_params.get('wine_tr_batch_size')
        tst_size = d_params.get('wine_test_batch_size')
        learning_rate = n_params.get('wine_lr')
        epochs = n_params.get('epochs')

        train_data = WineQuality(d_path, train= True, model= model_type, noise=noise, noise_type=noise_type, distribution_data = \
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


