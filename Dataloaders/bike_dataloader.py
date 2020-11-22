import glob
import os
import math
import random
from collections import defaultdict

import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

from settings import d_params
from utils import get_unif_Vmax, normalize_features, normalize_labels, get_dataset_stats, str_to_bool
from utils import generate_intervals, choose_distribution




class BikeSharing(Dataset):

    def __init__(self, path, seed = 42, train = True, noise = False , noise_type = None, \
                distribution_data = None, normalize = False, size = None):

        """
        Description:
            Bike Sharing Dataset (Fanaee-T & Gama, 2013) [#]_ is a dataset that contains 17,379 samples of the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information.
            
            The dataset has been processed, where the date was normalized from a scale between day 1 to day 730 to a scale between 0 and 4π. Then, the network is provided with the cosine and the sine of this number, which allowed to have the same representation for the same days of the year, while having the same distance between any two consecutive days,  keeping the cyclic nature of a year. 
            
            A similar idea was applied to hours,normalized from 0 to 2π instead of 0 to 24, and with the cosine and sine given to the network. The day of the week, being a category, was given as a one-hot vector of dimension 7. We also removed the season and the month as it was redundant information with the date.
        
        Attribute Information:
            The number of features after the preprocessing step is 19, which as following:

                             :f1:  Year
                             :f2-f4:  Date (sine and cos)
                             :f4-f5:  Hour (sine and cos)
                             :f6-f12:  Days of the week (one-hot vector)
                             :f13:  Holiday boolean
                             :f14:  Working day boolean
                             :f15:  Weather situation
                             :f16:  Temperature
                             :f17:  Felt temperature
                             :f18:  Humidity
                             :f19:  Wind speed


        Args:
            :path (string): A path to the UTKF dataset directory.
            :train (bool): A boolean that controls the selection of training data (True), or testing data (False).
            :noise (bool): A boolean that controls if the noise should be added to the data or not.
            :noise_type (string): A variable that controls the type of the noise.
            :distribution data (list): A list of information that is needed for noise generation.
            :normalize (bool): A boolean that controls if the data will be normalized (True) or not (False). 
            :size (int): Size of dataset (training or testing).
        
        """
        self.train = train
        self.data_path = path
        self.normalize = normalize
        self.noise = noise
        self.noise_type = noise_type
        self.dist_data = distribution_data
        self.data_slice_size = size
        self.seed = seed
        print("Train: {} -- data_slice_size: {}".format(train, size))
        # This is not the proper implementation, but doing that for research purproses.

        # Load the dataset
        self.features, self.labels = self.load_data()
        # Load the normalization constant variables.
        self.features_mean, self.features_std, self.labels_mean, self.labels_std = get_dataset_stats(dataset='BikeSharing')
        # Generate noise for the training samples.
        if self.train:
            if self.noise:
                self.lbl_noises, self.noise_variances, self.noisy_noise_variances = self.generate_noise(norm = self.normalize)  # normalize the noise. 
                # print("noises been added", self.lbl_noises)
                print("maximum noise", max(self.lbl_noises))
                # print("noise variances:", self.noise_variances)
                print("maximum noise variance:", max(self.noise_variances))
    
    
    def load_data(self):

        """
        Description:

            Loads the dataset.

        Return:
            features, labels.
        Return type:
            Tuple
        
        Args:
            None.
        """

        labels = pd.read_csv(os.path.join(self.data_path, 'labels.csv')).values.tolist()
        features = pd.read_csv(os.path.join(self.data_path, 'features.csv')).values.tolist()

        # Shuffling
        mapIndexPosition = list(zip(labels, features))
        random.seed(self.seed)
        random.shuffle(mapIndexPosition)
        labels, features = zip(*mapIndexPosition)


        if self.train:
            y = labels[:self.data_slice_size]
            x = features[:self.data_slice_size]

        else:
            y = labels[-self.data_slice_size:]
            x = features[-self.data_slice_size:]


        # Convert the data to numpy array
        x = torch.tensor(np.array(x),dtype=torch.float32)
        y = torch.tensor(np.array(y),dtype=torch.float32).squeeze(1)
        # set the length of the data to be the loaded data.
        self.data_length = len(y)
        print("train: {}, self.data_length: {}".format(self.train, self.data_length))

        return (x,y)



    
    def get_uniform_params(self, mu,v):
        """ Description:
            Generates the bounds of the uniform distribution using the mean and the variance, by solving the formula ::

                a = mu - sqrt(3*v)
                b = mu + sqrt(3*v)

        Return:
            Uniform distribution bounds a and b.
        
        Return type:
            Tuple.
        
        Args:
            :mu (float): The mean of the uniform distribution.
            :v  (float): The variance of the uniform distribution.
        """

        if v<0:
            raise ValueError(" Varinace is a negative number: {}".format(v))
        if mu<math.sqrt(3*v): # to prevent having negative values :), bc variance is a positive number.
            raise ValueError(" mu value is not valid, minimum value of mu: {}. Lower bound (a) will be negative, and that is not valid when variance is generated".format(math.sqrt(3*v)))
        else:
            a = mu - math.sqrt(3*v)
            b = mu + math.sqrt(3*v)

            return a,b
    

    def get_gamma_params(self,mu,v):
        """ Description:
            Generates the shape or concentration (alpha) and rate (beta) using the mean and variance of gamma distribution ::

                alpha = 1/k
                beta = 1/theta
                **
                k = (mu**2)/v
                theta = v/mu

        Return:
            Alpha, Beta .


        Return type:
            Tuple.

        Args:
            :mu (float): The mean of gamma distribution.
            :v  (float): The variance of gamma distribution.
        """
        
        if v == 0:
            raise ValueError("Variance is zaro, alpha and beta can not be estimated because of divideding by zero: {}".format(v))
        elif v <0:
            raise ValueError("Variance is a negative value: {}".format(v))
        
        elif v < mu**2:
            raise ValueError(" Variance should be greater than or equal to {}.".format(mu**2))
        else:
            theta = v/mu    # estimate the scale.
            k = (mu**2)/v   # estimate the shape.
            
            alpha = k       # estimate the shape
            beta = 1/theta  # estimate rate or beta

            return (alpha,beta)

    def get_distribution(self, dist_type, data, is_params_estimated, vmax=False, vmax_scale=1):
        """ Description:
            Create a probability distribution (uniform or gamma).

        Return:
            A probability distribution.

        Return type:
            Object.

        Args:
            :dist_type: An argument that specifies the type of the distribution.
            :data: A list that contains the information of distribution .
            :is_params_estimated: An argument that controls if the data is used used to create probability distribution. The data could be distribution statistics (mean and variance) or distribution parameters.
            :vmax: A boolean that controls if maximum heteroscedasticity will be used or not.
            :vmax_scale: An argument that specifies the heteroscedasticity scale.
        """

        noise_distributions = {}

        if is_params_estimated:
            print("Parameters are estimated")
            if dist_type =="uniform" or dist_type =="binary_uniform" :
                print("This is a "+ dist_type +  " distribution :)")
                for idx, param in enumerate(data):
                    if vmax:
                        v = get_unif_Vmax(param[0], scale_value=vmax_scale) 
                        a,b = self.get_uniform_params(param[0],v)
                    else:
                        a,b = self.get_uniform_params(param[0], param[1])

                    var_dist = torch.distributions.uniform.Uniform(a,b)
                    noise_distributions[str(idx+1)]=var_dist
        
            elif dist_type == "gamma":
                print("This is gamma distribution :)")
                for idx, param in enumerate(data):
                    alpha,beta = self.get_gamma_params(param[0], param[1])
                    var_dist = torch.distributions.gamma.Gamma(alpha,beta)
                    noise_distributions[str(idx+1)]=var_dist
        else:
            print("Parameters are not estimated")
            if dist_type =="uniform" or dist_type =="binary_uniform":
                print("This is uniform distribution :)")
                for idx, param in enumerate(data):
                    var_dist = torch.distributions.uniform.Uniform(param[0],param[1])
                    noise_distributions[str(idx+1)]=var_dist
        
            elif dist_type == "gamma":
                print("This is gamma distribution :)")
                for idx, param in enumerate(data):
                    var_dist = torch.distributions.gamma.Gamma(param[0],param[1])
                    noise_distributions[str(idx+1)]=var_dist

        return noise_distributions


    def gaussian_noise(self,var_dists,p=0.5):

        """ Description:
            Generates gaussian noises with a cenetred mean around 0 and heteroscedasticitical variance that sampled from a range of distributions.

        Return:
            Guassian noises and their heteroscedasticitical variances.


        Return type:
            Tuple.

        Args:
            :var_dist(object): Noise varaince probability distributions.
            :p (float): The contribution ratio of low and high noise variance distributions.
        """

        num_distributions = len(var_dists)
        boundaries = generate_intervals(num_distributions,p)
        noises_vars = []
        tracker = defaultdict(lambda : 0)
        for idx in range(self.data_slice_size):
            dist_id = choose_distribution(boundaries)
            var_distribution = var_dists.get(dist_id)
            noises_vars.append(var_distribution.sample((1,)))
            tracker[dist_id]+=1
           

        noise_dists_ratio = list(map(lambda x: (x[0],x[1]/self.data_slice_size*100), tracker.items())) 
        print("noise distributions ratio:", noise_dists_ratio)

        noises = [torch.distributions.normal.Normal(0, torch.sqrt(var)).sample((1,)).item() for var in noises_vars] 
        # for item in noises:
        #     if torch.isnan(torch.Tensor([item,])):
        #         print("NAN")
        # # print(noises_vars)
        
            
        return (noises, noises_vars)


        
    
    def generate_noise(self, norm = False):
        """
        Description:
            Unpacks information and calls gaussian_noise to generates noises.

        Return:
            Guassian noises and their heteroscedasticitical variances.

        Return type:
            Tuple.

        Args:
            :norm: Normalization.
        """

        dists = {}
        p = self.dist_data.get('coin_fairness')
        is_params_estimated = self.dist_data.get('is_params_est')
        data = self.dist_data.get("data")
        n = len(data)
        data = [(data[idx], data[idx+(n//2)]) for idx in range(n//2) ]
        

        if is_params_estimated:
            if self.noise_type == "uniform" or self.noise_type =="binary_uniform":
                is_vmax = self.dist_data.get("is_vmax")
                vmax_scale = self.dist_data.get("vmax_scale")
                dists = self.get_distribution(self.noise_type, data, is_params_estimated, is_vmax, vmax_scale)
            else:
                dists = self.get_distribution(self.noise_type, data, is_params_estimated)
        else:
            dists = self.get_distribution(self.noise_type, data, is_params_estimated)

        noises, noises_vars = self.gaussian_noise(dists, p)

        # apply noise on noise variance
        max_std_noise_var = torch.div(torch.FloatTensor(noises_vars),3)  # max std so that the probability is max 0.0015 to have a noisy variance less than 0 + noise is scaled by the variance
        eff_std_noise_var = torch.mul(max_std_noise_var, math.sqrt(self.dist_data.get("var_disturbance")))  # apply parameter (coefficient of proportionality for the variance w.r.t. max variance)
        noise_var_noise = [torch.distributions.normal.Normal(0, std).sample((1,)).item() for std in eff_std_noise_var]    # get noise
        noisy_noises_var = [torch.abs(noise_var_noise[i] + noises_vars[i]) for i in range(0, len(noise_var_noise))]   # use the absolute value to prevent negative variance (will only break mean(noise_variance_noise) = 0, that's why we have a max std to make it very unlikely)

        return (noises, noises_vars, noisy_noises_var)

            
    def __len__(self):

        """
        Description:
            return the length of the dataset (training or testing).

        Return:
            Dataset size

        Return Type:
            int.
        
        Args:
            None.
        """

        return self.data_length

    def __getitem__(self, idx):

        """
        Description:
            Sample batch of features, labels, noises and variances.

        Return:
            batch of features,  labels, noises, and noises variances ( training setting) or batch of features, labels ( testing setting),

        Return Type:
            Tuple.
        
        Args:
            :idx: auto-sampled generated index.
        """

        self.feature = self.features[idx]
        self.label = self.labels[idx]

    
        if self.train:
            if self.noise:
                self.label_noise = self.lbl_noises[idx]
                self.noise_variance = self.noise_variances[idx]
                self.noisy_noise_variance = self.noisy_noise_variances[idx]
                self.label = self.label + self.label_noise
                # Apply normalization to the noisy training data.
                if self.normalize:
                    self.feature = normalize_features(self.feature, self.features_mean, self.features_std,dataset='WineQuality')
                    self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                    # weight the noise value and its varince by the std of the labels of the dataset.
                    self.label_noise = self.label_noise/self.labels_std
                    self.noise_variance = self.noise_variance/(self.labels_std**2)
                    self.noisy_noise_variance = self.noisy_noise_variance/(self.labels_std**2)                
                return (self.feature, self.label, self.label_noise, self.noisy_noise_variance)
            else:
            # Apply normalization to the training data.
                if self.normalize:
                    self.feature = normalize_features(self.feature, self.features_mean, self.features_std,dataset='WineQuality')
                    self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)

                 
                return (self.feature, self.label)
        else:
            # Apply normalization to the testing data.
            if self.normalize:
                self.feature = normalize_features(self.feature, self.features_mean, self.features_std,dataset='WineQuality')
                self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                
            return (self.feature, self.label)  

        

