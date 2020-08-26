import glob
import os
import math
from collections import defaultdict

from PIL import Image
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset

from params import d_params
from utils import get_unif_Vmax, normalize_features, normalize_labels, get_dataset_stats, str_to_bool
from utils import generate_luck_boundaries, choose_luck_boundary, incremental_average
from utils import flip_coin



class UTKface(Dataset):

    def __init__(self, path, train = True, model="vanilla_cnn" ,transform = None, noise = False , noise_type = None, \
                distribution_data = None, normalize = False, noise_threshold = False, threshold_value = None):

        """
        Description:
            UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). 
            The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. 
            The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. 


        Args:
            :train (bool): Controls if the data will include the noise and its varaince with the actual inputs and labels or not.
            :train_size (int): Controls the number of samples in the training set.
            :images_path (string): Dataset directory path.
            :transform (Object): A torchvision transofrmer for processing the images.
            :noise_type (string): Controls the type of noise that will be added to the data
            :dist_data (tuple): A tuple of the mean and the variance of the noise distribution.
        
        """
        self.train = train
        self.data_paths = glob.glob(path)
        self.dataset_size  = len(self.data_paths)
        self.normalize = normalize
        self.model = model
        self.transform = transform
        self.noise = noise
        self.noise_type = noise_type
        self.dist_data = distribution_data
        self.train_size= d_params.get('train_size')
        self.noise_threshold = noise_threshold
        self.threshold_value = threshold_value
        
        # This is not the proper implementation, but doing that for research purproses.

        # Load the dataset
        self.images_path, self.labels = self.load_data()
        # Load the normalization constant variables.
        self.images_mean, self.images_std, self.labels_mean, self.labels_std = get_dataset_stats()
        # Generate noise for the training samples.
        if self.train:
            if self.noise:
                self.lbl_noises, self.noise_variances = self.generate_noise(norm = self.normalize)  # normalize the noise. 
                # print("noises been added", self.lbl_noises)
                print("maximum noise", max(self.lbl_noises))
                # print("noise variances:", self.noise_variances)
                print("maximum noise variance:", max(self.noise_variances))

                if self.noise_threshold:
                        print('Training data filtering started...')
                        self.images_path, self.labels, self.lbl_noises, self.noise_variances = self.filter_high_noise()
                        print('Training data filtering finished...')
    
    
    def filter_high_noise(self):
        filt_images_path = []
        filt_labels = []
        filt_lbl_noises = []
        filt_noise_variances = []
        filtered_data_counter = 0



        for idx in range(len(self.noise_variances)):
            if self.noise_variances[idx] < self.threshold_value:

                filt_images_path.append(self.images_path[idx])
                filt_labels.append(self.labels[idx])
                filt_lbl_noises.append(self.lbl_noises[idx])
                filt_noise_variances.append(self.noise_variances[idx])

                # Increase the counter value of the filtered data.
                filtered_data_counter = filtered_data_counter+1
            else:
                pass

        # set the length of the data to be the value of the filtered_data_counter.
        self.data_length = filtered_data_counter
        print('Number of filtered samples: {}'.format(filtered_data_counter))

        return (filt_images_path, filt_labels, filt_lbl_noises, filt_noise_variances)

    
    def load_data(self):

        """
        Description:

            Loads the dataset and preprocess it.

        Return:
            images paths and labels.
        Return type:
            Tuple
        
        Args:
            None.
        """
        
        labels = []

        if self.train:
            img_paths = self.data_paths[:self.train_size]
        else:
            img_paths = self.data_paths[self.train_size:]

        for path in img_paths:
            label = float(path.split("/")[-1].split("_")[0])
            labels.append(label)
            # set the length of the data to be the length of the train size.
        self.data_length = len(labels)
        return (img_paths,labels)



    
    def get_uniform_params(self, mu,v):
        """ Description:
            Generates the bounds of the uniform distribution by solving the formula ::

                a = mu - sqrt(3*v)
                b = mu + sqrt(3*v)

        Return:
            Uniform distribution bounds a and b.
        
        Return type:
            Tuple.
        
        Args:
            :mu (float): mean.
            :v  (float): variance.
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
            Alpha and Beta of gamma distribution.


        Return type:
            Tuple.

        Args:
            :mu (float): mean.
            :v  (float): variance.
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
            Create a distribution function from specified family by dist_type argument (uniform or gamma).

        Return:
            A distribution function.


        Return type:
            Object.

        Args:
            :std_dist(function): a distribution function for sampling the noises variances.
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
            Generates gaussian noises with mean 0 and heteroscedasticitical variance that sampled from one of a range of distributions (uniform or gamma).

        Return:
            Guassian noises and their heteroscedasticitical variances.


        Return type:
            Tuple.

        Args:
            :std_dist(object): a distribution function for sampling the noises variances.
        """
        num_distributions = len(var_dists)
        boundaries = generate_luck_boundaries(num_distributions,p)
        noises_vars = []
        tracker = defaultdict(lambda : 0)
        for idx in range(self.train_size):
            dist_id = choose_luck_boundary(boundaries)
            var_distribution = var_dists.get(dist_id)
            noises_vars.append(var_distribution.sample((1,)))
            tracker[dist_id]+=1
           

        noise_dists_ratio = list(map(lambda x: (x[0],x[1]/self.train_size*100), tracker.items())) 
        print("noise distributions ratio:", noise_dists_ratio)

        noises = [torch.distributions.normal.Normal(0, torch.sqrt(var)).sample((1,)).item() for var in noises_vars] 
            
        return (noises, noises_vars)


        
    
    def generate_noise(self, norm = False):
        """
        Description:
            Generates noises.

        Return:
            Guassian noises and their heteroscedasticitical variances.

        Return type:
            Tuple.

        Args:
            None.
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


        return (noises, noises_vars)

            
    def __len__(self):

        """
        Description:
            return the length of the dataset.

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
            Sample batch of images, labels, noises and variances.

        Return:
            batch of images, labels, noises (train setting) and variances (train setting).

        Return Type:
            Tuple.
        
        Args:
            :idx: auto-sampled generated index.
        """

        img_path = self.images_path[idx]
        self.label = self.labels[idx]

        # Convert the label to a tensor
        self.label = torch.tensor(self.label,dtype=torch.float32)
        self.image = Image.open(img_path)

        # Apply some transformation to the images.
        if self.transform:
            self.image = self.transform(self.image)

        if self.train:
            if self.noise:
                self.label_noise = self.lbl_noises[idx]
                self.noise_variance = self.noise_variances[idx]
                self.label = self.label + self.label_noise
                # Apply normalization to the noisy training data.
                if self.normalize:
                    self.image = normalize_features(self.image, self.images_mean, self.images_std)
                    self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                    # weight the noise value and its varince by the std of the labels of the dataset.
                    self.label_noise = self.label_noise/self.labels_std
                    self.noise_variance = self.noise_variance/(self.labels_std**2)

                return (self.image, self.label, self.label_noise, self.noise_variance)
            else:
            # Apply normalization to the training data.
                if self.normalize:
                    self.image = normalize_features(self.image, self.images_mean, self.images_std)
                    self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                return (self.image, self.label)
        else:
            # Apply normalization to the testing data.
            if self.normalize:
                self.image = normalize_features(self.image, self.images_mean, self.images_std)
                self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                
            return (self.image, self.label)  

        

