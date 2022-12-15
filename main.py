import os
import argparse

import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import  DataLoader
from torchvision import transforms

import wandb

# Import datasets
from Dataloaders.utkf_dataloader import UTKface
from Dataloaders.wine_dataloader import WineQuality
from Dataloaders.bike_dataloader import BikeSharing
from model import AgeModel, WineModel, BikeModel
from losses import BIVLoss, CutoffMSE
from train import Trainer

# Import default expirement settings

from settings import d_params
from settings import n_params
from settings import default_values

# Import helper tools
from utils import assert_args_mixture, print_experiment_information, str_to_bool, get_mean_avg_variance, get_dataset_stats


# Main 

if __name__ == "__main__":

    # Import default values
    distributions_ratio = default_values.get("distributions_ratio")
    maximum_hetero = default_values.get("maximum_hetero")
    hetero_scale = default_values.get("hetero_scale")
    epsilon = default_values.get("epsilon")
    threshold_value = default_values.get("threshold_value")
    warning_messages = default_values.get("warning_messages")


    # Parse arguments from the commandline
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")
    parser.add_argument("--experiment_settings", type=str, default="0")
    parser.add_argument("--model_settings", type=str,default="0")
    parser.add_argument("--noise_settings", type=str, default="0") 
    parser.add_argument("--params_settings", type=str, default="0")
    parser.add_argument("--parameters", type=str, default="0")
    parser.add_argument("--extra_exp", type=str, default="256,0,True,0.01")

    # Extract commandline arguments   
    args = parser.parse_args()
    experiment_settings = args.experiment_settings.split(",")
    model_settings = args.model_settings.split(",")
    noise_settings = args.noise_settings.split(",")
    params_settings = args.params_settings.split(",")
    parameters = args.parameters.split(",")
    extra_exp = args.extra_exp.split(",")

######################################################  Access CommandLine Arguments ############################################################

    # Access "experiment_settings" arguments
    tag = experiment_settings[0]
    seed = experiment_settings[1]
    assert isinstance(float(seed), float), "Argument: seed: " + warning_messages.get('datatype')
    seed = float(seed)
    dataset = experiment_settings[2]
    assert dataset in ["utkf","wine","bike"], "Argument: dataset: " + warning_messages.get('value')
    normalize = experiment_settings[3]
    assert isinstance( str_to_bool(normalize), bool), "Argument: normalize: " + warning_messages.get('bool')
    normalize = str_to_bool(normalize)
    train_size = experiment_settings[4]
    assert isinstance(int(train_size), int), "Argument: train_size: " + warning_messages.get('datatype')
    train_size = int(train_size)

    # Access "extra_experiments" arguments
    train_bsize = int(extra_exp[0])    # Batch size for training
    var_disturbance = float(extra_exp[1])  # Proportionality value for noise in the variance
    normalize_loss = str_to_bool(extra_exp[2])  # Boolean for the normalization of the weights in BIV loss function
    learning_rate = float(extra_exp[3])    # Learning rate


    # Access "model_settings" arguments
    model_type = model_settings[0]
    assert model_type in ["vanilla_ann","vanilla_cnn", "resnet"], "Argument: model_type: " + warning_messages.get('value')
    loss_type = model_settings[1]
    assert loss_type in ["mse", "cutoffMSE", "iv", "biv"], "Argument: loss_type: " + warning_messages.get('value') 

    if len(model_settings) > 2:
        if loss_type == "biv":
            epsilon = model_settings[2]
            assert epsilon.replace('.','',1).isdigit() , "Argument: epsilon: " + warning_messages.get('datatype')
            epsilon = float(epsilon)   
        elif loss_type == "cutoffMSE":
            threshold_value = model_settings[2]
            assert threshold_value.replace('.','',1).replace('-','',1).isdigit(), "Argument: threshold_value: " +  warning_messages.get('datatype')
            threshold_value = float(threshold_value)
            
            # Get labels's variance for normalizing the threshold value.
            if normalize:
                _,_,_, labels_std = get_dataset_stats(dataset)
                threshold_value = threshold_value/(labels_std**2)
        else:
            pass

    # Access noise settings

    noise = noise_settings[0] 
    assert isinstance( str_to_bool(noise), bool), "Argument: noise: " + warning_messages.get('bool')
    noise = str_to_bool(noise)
    if noise:
        noise_type = noise_settings[1]
        assert noise_type in ["binary_uniform","uniform","gamma"], "Argument: noise_type: " + warning_messages.get('value')

  
  # Access parameters settings

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
                if noise_type == "binary_uniform":
                    hetero_scale = parameters[3]
                elif noise_type == "uniform":
                    hetero_scale = parameters[2]
                assert float(hetero_scale)>=0 and float(hetero_scale)<=1 , "Argument: hetero_scale: "+ "argument value is not recognized."
                hetero_scale = float(hetero_scale)
            parameters = parameters[1:]
        
        # Get distributions parameters
        for item in parameters: assert item.replace('.','',1).replace('-','',1).isdigit() , "Argument: parameters: " + "datatype is not supported."
        parameters = list(map(lambda x: float(x), parameters))
    else:
        noise_type = None
        is_estim_noise_params = False
        params_type = None
        parameters = None

    
   # Print experiments information
    arguments = {"tag": tag, "seed": seed, "dataset": dataset, "normalize": normalize, "train_size": train_size, "loss_type": loss_type, "learning_rate": learning_rate, "model_type": model_type, 
                 "noise": noise, "noise_type": noise_type, "is_estim_noise_params": is_estim_noise_params, "epsilon": epsilon, "threshold value": threshold_value, 'params_type':params_type,
                 'parameters':parameters, 'train_batch_size': train_bsize, "var_disturbance": var_disturbance, "normalize_loss": normalize_loss}
    
    # Print Experiment Information
    print_experiment_information(arguments)
    # Apply complex assertions.
    assert_args_mixture(arguments)


############################################################### Run the experiment ##############################################################
   
    # Get Wandb tags
    tag = [tag,]
    # Initiate wandb client.
    wandb.init(project="iv_deep_learning",tags=tag , entity="montreal_robotics", config=arguments)
    # Get the api key from the environment variables.
    api_key = os.environ.get('WANDB_API_KEY')
    # login to my wandb account.
    wandb.login(api_key)
   
   
    # Set expirement seed
    torch.manual_seed(seed)
    # Set experiment id
    exp_id = params_type

    
    # Prepare distribution data to be passed to the dataloader.

    
    if noise and noise_type =="binary_uniform" and params_type=="meanvar_avg":
        # Overwrite mu2 if the average noise variance (X) is passed in the commandline arguments.
        parameters[1] = get_mean_avg_variance(noise_type,average_variance,parameters[0],distributions_ratio)
        dist_data = {"coin_fairness":distributions_ratio,"is_params_est":is_estim_noise_params, "is_vmax":maximum_hetero, "vmax_scale":hetero_scale ,"data":parameters, "var_disturbance":var_disturbance}

    elif noise and noise_type == "binary_uniform":
        dist_data = {"coin_fairness":distributions_ratio, "is_params_est":is_estim_noise_params,"is_vmax":maximum_hetero, "vmax_scale":hetero_scale, "data":parameters, "var_disturbance":var_disturbance}
    else:
        dist_data = {"coin_fairness":distributions_ratio, "is_params_est":is_estim_noise_params,"is_vmax":maximum_hetero, "vmax_scale":hetero_scale,"data":parameters, "var_disturbance":var_disturbance}

    # Define the dataset
    if dataset == "utkf":

        d_path = d_params.get('d_path')
        tr_size = d_params.get('tr_batch_size')
        if train_bsize:
            tr_size = train_bsize
        tst_size = d_params.get('test_batch_size')
        #learning_rate = n_params.get('lr')
        epochs = n_params.get('utkf_epochs')
        test_size = d_params.get('test_size')
        dataset_size = d_params.get('dataset_size')
        assert test_size+train_size<=dataset_size, warning_messages.get("CustomMess_dataset").format(train_size, test_size, dataset_size)
        
        trans= torchvision.transforms.Compose([transforms.ToTensor()])

        train_data = UTKface(d_path, transform= trans, train= True, noise=noise, noise_type=noise_type, distribution_data = \
                     dist_data, normalize=normalize, size=train_size) 
        
        test_data = UTKface(d_path, transform= trans, train= False,  normalize=normalize, size=test_size)


    elif dataset == "wine":

        d_path = d_params.get('wine_path')
        tr_size = d_params.get('wine_tr_batch_size')
        if train_bsize:
            tr_size = train_bsize
        tst_size = d_params.get('wine_test_batch_size')
        #learning_rate = n_params.get('wine_lr')
        epochs = n_params.get('wine_epochs')
        test_size = d_params.get('wine_test_size')
        dataset_size = d_params.get('wine_dataset_size')
        assert test_size+train_size<=dataset_size, warning_messages.get("CustomMess_dataset").format(train_size, test_size, dataset_size)

        train_data = WineQuality(d_path, train= True,  noise=noise, noise_type=noise_type, distribution_data = \
                                        dist_data, normalize=normalize, size=train_size) 
        test_data = WineQuality(d_path, train= False,  normalize=normalize, size=test_size)

       
    elif dataset == "bike":

        d_path = d_params.get('bike_path')
        tr_size = d_params.get('bike_tr_batch_size')
        if train_bsize:
            tr_size = train_bsize
        tst_size = d_params.get('bike_test_batch_size')
        #learning_rate = n_params.get('bike_lr')
        epochs = n_params.get('bike_epochs')
        test_size = d_params.get('bike_test_size')
        dataset_size = d_params.get('bike_dataset_size')
        assert test_size+train_size<=dataset_size, warning_messages.get("CustomMess_dataset").format(train_size, test_size, dataset_size)

        train_data = BikeSharing(d_path, seed=seed, train= True,  noise=noise, noise_type=noise_type, distribution_data = \
                                        dist_data, normalize=normalize, size=train_size) 
        test_data = BikeSharing(d_path, seed=seed, train= False,  normalize=normalize, size=test_size)

    # Load the data
    print("Training batch size: {}".format(tr_size))

    train_loader = DataLoader(train_data, batch_size=tr_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=tst_size, drop_last=True, shuffle=True)

    # Select the model
    if model_type =="vanilla_ann" and dataset=='wine':
        model = WineModel()
        print("#"*80,"Model is:{}".format(model_type), "#"*80)
    elif model_type =="vanilla_ann" and dataset=='bike':
        model = BikeModel()
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

    # Select the loss function
    if loss_type == "biv":
        loss = BIVLoss(epsilon=epsilon, normalize = normalize_loss)
    elif loss_type == "cutoffMSE":
        loss = CutoffMSE(cutoffValue=threshold_value)
    else:
        loss = torch.nn.MSELoss()


    # Trainer
    trainer = Trainer(experiment_id=tag[0], train_loader= train_loader, test_loader= test_loader, \
        model=model, loss= loss, optimizer= optimz, epochs = epochs)

    # Call wandb to log model performance.
    wandb.watch(model)

    # train the model
    trainer.train(loss_type=loss_type)


