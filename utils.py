import pandas as pd

import torch

import wandb

def get_unif_Vmax(mu, scale_value =1):
     
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
    vmax = vmax/scale_value

    return vmax


def get_dataset_stats(xtrain, ytrain):

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

def normalize_images(images, images_mean, images_std):
    
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
        # print(binq)
        # print("+++++++++++++++")
        # Reshape the images 
        x = torch.stack(x.values.tolist()).cuda(0)
        y = torch.unsqueeze(torch.Tensor(y.values.tolist()),1).cuda(0)
        out = model(x)
        gloss = loss(out,y)

        wandb.log({'bin loss':gloss.item()}, step=i)

    return 0
