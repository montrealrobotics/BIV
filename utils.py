import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch

import wandb

from settings import d_params, n_params

def get_unif_Vmax(mu, scale_value=1):
     
    """
    Description:
        Estimates the maximum variance needed for the uniform distribution to produce maximum heteroscedasticity.

    Return:
        :vmax: the maximum uniform variance.
    Return type:
        float
    Args:
        :mu: mean of the the uniform.
        :scale_value [optional]: controls scaling the Vmax to different values.
    
    Here is the formula for estimating :math:`V_{max}`

    .. math::
        V_{max} = \\frac{4 \mu^2}{12}

    """
    vmax = (4*mu**2)/12
    vmax = vmax*scale_value

    return vmax



def get_dataset_stats(dataset='UTKFace'):

         
    """
    Description:
        Gets dataset statistics. The statistics include:

            1) Features mean.
            2) Features std.
            3) Labels mean.
            4) Labels std.


    Return:
        :features_mean: mean of the input features.
        :features_std: standard deviation of the input features.
        :labels_mean: mean of the labels.
        :labels_std: standard deviation of the labels.
    Return type:
        Tuple.
    Args:
        :dataset: Dataset name.
    

    """


    if dataset == 'UTKFace' or dataset == 'utkf':
        images_mean = torch.Tensor(pd.read_csv(d_params['d_img_mean_path']).values)[0][1]
        images_std = torch.Tensor(pd.read_csv(d_params['d_img_std_path']).values)[0][1]
        
        labels_mean = torch.Tensor(pd.read_csv(d_params['d_lbl_mean_path']).values)[0][1]
        labels_std = torch.Tensor(pd.read_csv(d_params['d_lbl_std_path']).values)[0][1]

        return ( images_mean, images_std, labels_mean, labels_std ) 
    elif dataset=='WineQuality' or dataset == 'wine': 

        features_mean = np.genfromtxt(d_params['wine_features_mean_path'], delimiter=',') 
        features_std = np.genfromtxt(d_params['wine_features_std_path'], delimiter=',') 
        labels_mean = np.genfromtxt(d_params['wine_lbl_mean_path'],delimiter=',')
        labels_std = np.genfromtxt(d_params['wine_lbl_std_path'], delimiter=',')

        features_mean = torch.tensor(features_mean,dtype=torch.float32)
        features_std = torch.tensor(features_std,dtype=torch.float32)

        labels_mean = torch.tensor(labels_mean,dtype=torch.float32)
        labels_std = torch.tensor(labels_std,dtype=torch.float32)

        return ( features_mean, features_std, labels_mean, labels_std ) 
    elif dataset=='BikeSharing' or dataset == 'bike': 

        features_mean = np.genfromtxt(d_params['bike_features_mean_path'], delimiter=',') 
        features_std = np.genfromtxt(d_params['bike_features_std_path'], delimiter=',') 
        labels_mean = np.genfromtxt(d_params['bike_lbl_mean_path'],delimiter=',')
        labels_std = np.genfromtxt(d_params['bike_lbl_std_path'], delimiter=',')

        features_mean = torch.tensor(features_mean,dtype=torch.float32)
        features_std = torch.tensor(features_std,dtype=torch.float32)

        labels_mean = torch.tensor(labels_mean,dtype=torch.float32)
        labels_std = torch.tensor(labels_std,dtype=torch.float32)

        return ( features_mean, features_std, labels_mean, labels_std ) 
    else:
        raise ValueError('Dataset is not recognized.')

        

def normalize_labels(labels, labels_mean, labels_std):
    
    """
    Description:
        Normalize the training labels as follow:

    .. math::
        \widetilde{y} = \\frac{(y - \overline{y})} {\sigma_y}.
    
    where

    .. math::
        \widetilde{y} = \\text{Normalized labels, } 
        y = \\text{labels, }
        \overline{y} = \\text{Mean of the labels, }.
        \sigma = \\text{Standard deviation of the labels}
    Return:
        :labels_norm: Normalized labels.
    Return type:
        1D Tensor.
    Args:
        :labels: Training labels.
        :labels_mean: Emprical mean of the training labels .
        :labels_std: Emprical standard deviation of the training labels.
    """

    labels_norm = (labels - labels_mean)/labels_std
    return labels_norm


def normalize_features(features, features_mean, features_std, dataset='UTKFace'):
    
    """
    Description:
        Normalize the training input features as follows:

    .. math::
        \widetilde{X} = \\frac{(X - \overline{X})} {\sigma_x}.
    
    where

    .. math::
        \widetilde{X} = \\text{Normalized features, } 
        X = \\text{Input features, }
        \overline{X} = \\text{Features mean, }.
        \sigma = \\text{Features standard deviation}
    Return:
        :images_norm: Normalized Features.
    Return type:
        NxD Tensor
    Args:
        :features: train data. (images)
        :features_mean: Mean of the features.
        :features_std:  Standard deviation of the features.
        :dataset: Name of the dataset that needs to be normalized.
    """

    if dataset == 'UTKFace':
        channels = features.shape[0]
        width = features.shape[1]
        length = features.shape[2]
        
        features = features.squeeze().view(1,-1) # rolling out the whole training dataset to be a one vector.
        
        features_norm = (features - features_mean) / features_std
        
        # reshape the image
        features_norm = features_norm.view(channels,length,width)
        return features_norm
    elif dataset=="WineQuality" or dataset=="BikeSharing":
        features_norm = (features - features_mean) / features_std
        return features_norm
    else:
        raise ValueError("Dataset is not recognized.")


def str_to_bool(string):
    """
    Description:
        Convert a string to a boolean.
    Return:
        True or False.
    Return type:
        boolean
    Args:
        :string: A string to be converted.
    
    """
    if string =='True':
        return True
    elif string == 'False':
        return False
    else:
        if isinstance(string, str) :
            raise ValueError("Received {} as an argument. Only 'True' or 'False' are accepted.".format(string))
        else:
            raise TypeError("The argument is not a string but a {}.".format(type(string)))



def generate_intervals(num_dists, p=0.5):

    """
    Description:
        Generates intervals for sampling noise variance distributions. The intervals are low, which assigns low sampling probablity values, and high, that assigns high values.
    Return:
        intervals 
    Return type:
        Dictionary
    Args:
        :num_dists: Number of distributions that need to have sampling intervals.
        :p: Interval balance factor.It controls the balance between low and high intervals.
    
    """

    if num_dists ==0:
        raise ValueError(" number of distributions are zero: {}".format(num_dists))
    elif num_dists ==1:
        p =1
        l = np.linspace(0,p,num_dists+1, endpoint=True)
    else:
        l1 = np.linspace(0,p,(num_dists//2)+num_dists%2, endpoint=False)
        l2 = np.linspace(p,1,(num_dists//2)+1, endpoint=True,)
        l = np.concatenate((l1,l2))

    intervals = {}

    for i in range(len(l)-1):
        intervals[str(i+1)] = (l[i], l[i+1])
    
    return intervals


def choose_distribution(intervals):
    """
    Description:
        Chooses a distribution based on sampling from the intervals.
    Return:
        key, (an id for the sampled distribution)
    Return type:
        int
    Args:
        :intervals: A dictionary of tuple values, each tuple represents an interval that is compared against when the sampling happens.
        :p: Interval balance factor.It controls the balance between low and high intervals.
    """

    random_number = torch.rand((1,1)).item()
    for key in intervals.keys():
        if random_number> intervals.get(key)[0] and random_number< intervals.get(key)[1]:
            return key
    
    raise RuntimeError("generated random number does not fall into any category: {}".format(random_number))




def get_mean_avg_variance(noise_type,avg_variance_m,mu1,p):
    """
    Description:
         Estimates the value of the mean of the second distribution, in a bi-model distribution, based on the average noise varaince mean.
    Return:
        mu2
    Return type:
        float
    Args:
        :noise_type: Noise type.
        :avg_variance_m: The average of means of the noise variance distributions.
        :mu1: First distribution's mean.
        :p: Distributions ratio.

    
    """

    mu2 = (avg_variance_m-p*mu1)/(1-p)
    return mu2



def print_experiment_information(args):

    """
    Description:
         Print experiment information, which consists of :

            1) Dataset information.
            2) Models informations.
            3) Commandline options information.
    Return:
        None
    Return type:
        None
    Args:
        :args: Commandline options.

    
    """

    print("#"*80,"Dataset Settings:","#"*80)
    print(d_params)
    print("#"*80,"Network Settings:","#"*80)
    print(n_params)
    print("#"*80,"CommandLine Arguments:","#"*80)
    print(args)
    print("*"*180)


def filter_batch(predictions, labels, noise_var, threshold = 0.4):
    """
    Description:
         Filters a batch of labels based on a noise varaince threshold.
    Return:
        filtered predictions, filtered labels , filtered noise_variance 
    Return type:
        Tuple
    Args:
        :predictions: model's predictions.
        :labels: Noisy labels.
        :noise_var: noises variances 
        :threshold: noise variance threshold.

    
    """

    noise_var_mask = noise_var < threshold
    noise_var = noise_var[noise_var_mask]
    labels_n = labels[noise_var_mask]
    predictions_n = predictions[noise_var_mask]

    num_filtered_samples = len(labels_n)
    num_total_samples = len(labels)
    print("Number of filtered samples per batch: {}".format(num_filtered_samples))
    print("Ratio of filtered samples per batch: {}".format((num_filtered_samples/num_total_samples)*100))

    return predictions_n, labels_n , noise_var 




def assert_args_mixture(args):
    """
    Description:
         Check the validaty of the combination of two or more arguments.
    Return:
        None
    Return type:
        None
    Args:
        :args: Arguments that need to be checked.
    """

    if args.get('loss_type') == "biv" and args.get('noise') ==False:
        raise RuntimeError("BIV needs noise variance to work properly. Please enable 'noise'= True and specifiy the noise.")

    
    return None