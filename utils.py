import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import torch

import wandb

from params import d_params, n_params

def get_unif_Vmax(mu, scale_value=1):
     
    """"
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


def compute_dataset_stats(xtrain, ytrain):

    """"
    Description:
        Estimates the mean and the variance for the images and their labels.

    Return:
        :stats: A tuple of the mean and the variance of the images and their corressponding labels.
    Return type:
        Tuple
    Args:
        :xtrain: train data. (images)
        :ytrain: train labels.

    """
    
    xtrain_mean = xtrain.squeeze().view(1,-1).mean()
    xtrain_std = xtrain.squeeze().view(1,-1).std()
    
    ytrain_mean = ytrain.mean()
    ytrain_std = ytrain.std()

    
    pd.DataFrame((ytrain_mean.numpy(),)).to_csv("labels_mean.csv")
    pd.DataFrame((ytrain_std.numpy(),)).to_csv("labels_std.csv")
    
    pd.DataFrame((xtrain_mean.numpy(),)).to_csv("images_mean.csv")
    pd.DataFrame( (xtrain_std.numpy(),)).to_csv("images_std.csv")
    
    return (xtrain_mean, xtrain_std, ytrain_mean, ytrain_std)
    


def get_dataset_stats(dataset='UTKFace'):

    if dataset == 'UTKFace':
        images_mean = torch.Tensor(pd.read_csv(d_params['d_img_mean_path']).values)[0][1]
        images_std = torch.Tensor(pd.read_csv(d_params['d_img_std_path']).values)[0][1]
        
        labels_mean = torch.Tensor(pd.read_csv(d_params['d_lbl_mean_path']).values)[0][1]
        labels_std = torch.Tensor(pd.read_csv(d_params['d_lbl_std_path']).values)[0][1]

        return ( images_mean, images_std, labels_mean, labels_std ) 
    elif dataset=='WineQuality': 

        features_mean = np.genfromtxt(d_params['wine_features_mean_path'], delimiter=',') 
        features_std = np.genfromtxt(d_params['wine_features_std_path'], delimiter=',') 
        labels_mean = np.genfromtxt(d_params['wine_lbl_mean_path'],delimiter=',')
        labels_std = np.genfromtxt(d_params['wine_lbl_std_path'], delimiter=',')

        features_mean = torch.tensor(features_mean,dtype=torch.float32)
        features_std = torch.tensor(features_std,dtype=torch.float32)

        labels_mean = torch.tensor(labels_mean,dtype=torch.float32)
        labels_std = torch.tensor(labels_std,dtype=torch.float32)

        return ( features_mean, features_std, labels_mean, labels_std ) 
    else:
        raise ValueError('Dataset is not recognized.')

        



def normalize_labels(labels, labels_mean, labels_std):
    
    """"
    Description:
        Normalize the training labels as follow:
    .. math::
        normalized labels = (labels - labels mean) / labels standard deviation.
    Return:
        :labels_norm: Normalized labels.
    Return type:
        Tensor
    Args:
        :labels: train labels. (images)
        :labels_mean: labels mean.
        :labels_std: labels standard deviation.
    """

    labels_norm = (labels - labels_mean)/labels_std
    return labels_norm


def normalize_features(features, features_mean, features_std, dataset='UTKFace'):
    
    """"
    Description:
        Normalize the training images as follow:
    .. math::
        normalized images = (images - images mean) / images standard deviation.
    Return:
        :images_norm: Normalized images.
    Return type:
        Tensor
    Args:
        :images: train data. (images)
        :images_mean: images mean.
        :images_std: images standard deviation.
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
    elif dataset=="WineQuality":
        features_norm = (features - features_mean) / features_std
        return features_norm
    else:
        raise ValueError("Dataset is not recognized.")

    

def normalize_images_batch(images, images_mean, images_std):
    
    """"
    Description:
        Normalize the training images as follow:
    .. math::
        normalized images = (images - images mean) / images standard deviation.
    Return:
        :images_norm: Normalized images.
    Return type:
        Tensor
    Args:
        :images: train data. (images)
        :images_mean: images mean.
        :images_std: images standard deviation.
    """
    # print(images.shape)
    batch_size = images.shape[0]
    channels = images.shape[1]
    width = images.shape[2]
    length = images.shape[3]
    
    images = images.squeeze().view(1,-1) # rolling out the whole training dataset to be a one vector.
    # print("##################################")
    # print(images.shape)
    # print(images_mean.shape)
    # print(images_mean.shape)
    # print(images_std.shape)
    
    images_norm = (images - images_mean) / images_std
    
    # reshape the image
    images_norm = images_norm.view(batch_size,channels,length,width)
    
    
    return images_norm


def group_labels(x,y,b_num):
    
    # Convert torch tensor to pandas's dataframe
    x_pd = pd.DataFrame(x,columns=['x',])
    y_pd = pd.DataFrame(y,columns=['y',])
    # Discretize the data into groups based on bining operation.
    labels_names =  ['b'+str(i+1) for i in range(b_num) ] 
    y_b = pd.qcut(pd.DataFrame(y)[0],b_num, labels =labels_names)
    # Convert from pandas's category data type to dataframe datatype.
    y_b = pd.DataFrame(y_b.values, columns=['bin']) 
    # Concatenate x_pd,y_pd, y_b

    data = pd.concat([x_pd, y_pd, y_b],axis=1)
    
    # group the data based on bins
    groups = []
    for b in range(b_num):
        groups.append(data[data['bin']== labels_names[b]])
    
    return groups


def group_testing(x_pred, y_pred, bin_num, model, loss):

    groupsBybin = group_labels(x_pred,y_pred,bin_num)
    
    for i in range(bin_num):
        group = groupsBybin[i]
        x = group['x']
        y = group['y']
        binq =group['bin']
        # Reshape the images 
        x = torch.stack(x.values.tolist()).cuda(0)
        y = torch.unsqueeze(torch.Tensor(y.values.tolist()),1).cuda(0)
        out = model(x)
        gloss = loss(out,y)

        wandb.log({'bin loss':gloss.item()}, step=i)

    return 0


def str_to_bool(arg):
    if arg =='True':
        return True
    elif arg == 'False':
        return False
    else:
        if isinstance(arg, str) :
            raise ValueError("Received {} as an argument. Only 'True' or 'False' are accepted.".format(arg))
        else:
            raise TypeError("The argument is not a string but a {}.".format(type(arg)))

def plot_hist(x, name):
    plot_path = '/final_outps/'+str(name)+'.png'
    plt.hist(x)
    plt.savefig(plot_path)
    wandb.save(plot_path)
    
    return 0



def flip_coin(p =0.5):
    random_number = torch.rand(size=(1,1))

    if random_number <p:
        return "low"
    else:
        return "high"


def generate_luck_boundaries(num_dists, p=0.5):
    if num_dists ==0:
        raise ValueError(" number of distributions are zero: {}".format(num_dists))
    elif num_dists ==1:
        p =1
        l = np.linspace(0,p,num_dists+1, endpoint=True)
    else:
        l1 = np.linspace(0,p,(num_dists//2)+num_dists%2, endpoint=False)
        l2 = np.linspace(p,1,(num_dists//2)+1, endpoint=True,)
        l = np.concatenate((l1,l2))

    boundaries = {}

    for i in range(len(l)-1):
        boundaries[str(i+1)] = (l[i], l[i+1])
    
    return boundaries


def choose_luck_boundary(boundaries):
    random_number = torch.rand((1,1)).item()
    for key in boundaries.keys():
        if random_number> boundaries.get(key)[0] and random_number< boundaries.get(key)[1]:
            return key
    
    raise RuntimeError("generated random number does not fall into any category: {}".format(random_number))



def incremental_average(x,x_old, i):
    average = x_old + (x-x_old)*1/(i+1)
    return average

def incremental_average_full(x):
    x_old =0
    for i, e in enumerate(x):
        average = x_old + (e-x_old)*1/(i+1)
        x_old = average
    
    return x_old


def average_noise_mean(noise_type,avg_m,mu1,p):
    # if noise_type == "gamma":
    #     print(v2, v2**2)
    #     condition = np.sqrt(v2) *(1-p) + p*mu1
    #     print("QQQ", condition)
    #     if avg_m > condition:
    #         raise ValueError(" Noise variance average should be less or equal to {}".format(condition))

    mu2 = (avg_m-p*mu1)/(1-p)
    return mu2



def assert_arguments(arguments):

    messages = {"bool":":argument is not boolean.", "datatype":"datatype is not supported.", "value":"argument value is not recognized."}


    assert isinstance(float(arguments.get('seed')), float), "Argument: seed: " + messages.get('datatype')
    assert arguments.get('dataset') in ["utkf","wine"], "Argument: dataset: " + messages.get('value')
    assert isinstance( str_to_bool(arguments.get('normalize')), bool), "Argument: normalize: " + messages.get('bool')
    assert arguments.get('loss_type') in ["mse", "iv", "biv"], "Argument: loss_type: " + messages.get('value')
    assert arguments.get('model_type') in ["vanilla_ann","vanilla_cnn", "resnet"], "Argument: model_type: " + messages.get('value')
    # assert arguments.get('average_variance').replace('.','',1).replace('-','',1).isdigit(), "Argument: average_variance: "+ messages.get('value')

    assert isinstance( str_to_bool(arguments.get('noise')), bool), "Argument: noise: " + messages.get('bool')
    assert arguments.get('noise_type') in ["binary_uniform","uniform","gamma"], "Argument: noise_type: " + messages.get('value')
    assert isinstance( arguments.get('is_estim_noise_params'), bool), "Argument: estimate_noise_params: " + messages.get('bool')

    assert arguments.get('params_type') in ["meanvar","meanvar_avg","boundaries","alphabeta"], "Argument: params_type: " + messages.get('value')

    # assert float(arguments.get('distributions_ratio'))>=0 and float(arguments.get('distributions_ratio'))<=1 , "Argument: distributions_ratio: "+ messages.get('value')
    
    # Handle distributions parameters

    # for item in arguments.get('parameters'): assert item.replace('.','',1).replace('-','',1).isdigit() , "Argument: parameters: " + messages.get('datatype')

    return 0



def print_experiment_information(args):

    print("#"*80,"Dataset Settings:","#"*80)
    print(d_params)
    print("#"*80,"Network Settings:","#"*80)
    print(n_params)
    print("#"*80,"CommandLine Arguments:","#"*80)
    print(args)
    print("*"*180)


def filter_batch(predictions, labels, noise_var, threshold = 0.4):
    noise_var_mask = noise_var < threshold
    noise_var = noise_var[noise_var_mask]
    labels_n = labels[noise_var_mask]
    predictions_n = predictions[noise_var_mask]

    num_filtered_samples = len(labels_n)
    num_total_samples = len(labels)
    print("Number of filtered samples per batch: {}".format(num_filtered_samples))
    print("Ratio of filtered samples per batch: {}".format((num_filtered_samples/num_total_samples)*100))

    return predictions_n, labels_n , noise_var 


def filter_batch_v2(batch, labels, noise_var, threshold = 0.4):
    batch_arr = []
    label_arr = []
    variance_arr = []
    count = 0

    for i in range(len(noise_var)):
        if noise_var[i]< threshold:
            batch_arr.append(batch[i])
            label_arr.append(labels[i])
            variance_arr.append(noise_var[i])
            count+=1
    
    batch = torch.stack(batch_arr)
    labels = torch.tensor(label_arr)
    variances = torch.tensor(variance_arr)

    # Handle the dimension for 
    labels = torch.unsqueeze(labels, 1)
    variances = torch.unsqueeze(variances, 1)

    print("Number of filtered samples per batch: {}".format(count))
    print("Ratio of filtered samples per batch: {}".format((count/len(noise_var))*100))
    return batch, labels , variances 



def assert_args_mixture(args):
        # arguments = {"tag": tag, "seed": seed, "dataset": dataset, "normalize": normalize, "train_size": train_size, "loss_type": loss_type, "model_type": model_type, 
        #          "noise": noise, "noise_type": noise_type, "is_estim_noise_params": is_estim_noise_params, 'params_type':params_type,
        #          'parameters':parameters}

    if args.get('loss_type') == "biv" and args.get('noise') ==False:
        raise RuntimeError("BIV needs noise variance to work properly. Please enable 'noise'= True and specifiy the noise.")

    
    return 0