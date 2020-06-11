import glob
import os
import math

from PIL import Image
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset

from params import d_params
from utils import get_unif_Vmax, normalize_images, normalize_labels, get_dataset_stats, str_to_bool


class UTKface(Dataset):

    def __init__(self, path, train = True, transform = None, noise = False , noise_type = None, distribution_data = None, normalize = False):

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
        self.transform = transform
        self.noise = noise
        self.noise_type = noise_type
        self.dist_data = distribution_data
        self.train_size= d_params.get('train_size')
        

        # This is not the proper implementation, but doing that for research purproses.

        # Load the dataset
        self.images_pth, self.labels = self.__load_data()
        # Load the normalization constant variables.
        self.images_mean, self.images_std, self.labels_mean, self.labels_std = get_dataset_stats()
        # Generate noise for the training samples.
        if self.train:
            if self.noise:
                if self.noise_type == 'uniform':
                    self.lbl_noises, self.noise_variances = self.generate_noise(norm = self.normalize)  # normalize the noise.  
                else:
                    print("Exception: you must specify a noise, either 'uniform' or 'gauss' ")





        
    
    def __load_data(self):

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
        train_labels = []
        if self.train:
            train_img_paths = self.data_paths[:self.train_size]
            for path in train_img_paths:
                label = path.split("/")[3].split("_")[0]
                train_labels.append(int(label))

            # set the length of the dataset to be the length of the test size.
            self.data_length = len(train_labels)
            return (train_img_paths,train_labels)

        else:
            test_labels = []
            test_img_paths = self.data_paths[self.train_size:]
            for path in test_img_paths:
                label = path.split("/")[3].split("_")[0]
                test_labels.append(int(label))

            # set the length of the dataset to be the length of the test size.
            self.data_length = len(test_labels)
            return (test_img_paths,test_labels)

    
    def get_unif_bounds(self, mu,v):
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
        b = mu + math.sqrt(3*v)
        a = mu - math.sqrt(3*v)

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
        
        theta = v/mu    # estimate the scale.
        k = (mu**2)/v   # estimate the shape.
        
        alpha = k       # estimate the shape
        beta = 1/theta  # estimate rate or beta

        return (alpha,beta)


    def get_distribution(self, dist_type, mu, v):
        """ Description:
            Create a distribution function from specified family by dist_type argument (uniform or gamma).

        Return:
            A distribution function.


        Return type:
            Object.

        Args:
            :std_dist(function): a distribution function for sampling the noises variances.
        """

        if dist_type =="uniform":
            a,b = self.get_unif_bounds(mu, v)
            std_dist = torch.distributions.uniform.Uniform(a,b)

        
        elif dist_type == "gamma":
            alpha, beta = self.get_gamma_params(mu,v)
            std_dist = torch.distributions.gamma.Gamma(alpha,beta)

        return std_dist


    def gaussian_noise(self, std_dist, normalize = False):

        """ Description:
            Generates gaussian noises with mean 0 and heteroscedasticitical variance that sampled from one of a range of distributions (uniform or gamma).

        Return:
            Guassian noises and their heteroscedasticitical variances.


        Return type:
            Tuple.

        Args:
            :std_dist(object): a distribution function for sampling the noises variances.
        """
            
        # Sample heteroscedasticitical noises stds for the whole training set.
        noises_stds = std_dist.sample((self.train_size,))

        # Sample gaussian noises.
    
        if normalize:
            noises_stds = noises_stds/self.labels_std
   

        noises = [torch.distributions.normal.Normal(0, std).sample((1,)).item() for std in noises_stds]

        return (noises, noises_stds)


    
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

        if self.noise_type == "uniform":
        
            mu = self.dist_data[0]
            scale = str_to_bool(self.dist_data[2]) # True or False
            print("###################### Debug #########################")
            print("Vmax Scale: ", scale)
            if scale:
                v = get_unif_Vmax(mu, scale_value=self.dist_data[3])
            else:
                v =  self.dist_data[1]
        
        elif self.noise_type == "gamma":
            mu = self.dist_data[0]
            v =  self.dist_data[1]
 

        dist = self.get_distribution(self.noise_type,mu,v)
        lbl_noises, noise_variances = self.gaussian_noise(dist, normalize = norm)

        return (lbl_noises, noise_variances)

            
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

        img_path = self.images_pth[idx]
        self.label = self.labels[idx]

        # Convert the label to a tensor
        self.label = torch.tensor(self.label,dtype=torch.float32)
        self.image = Image.open(img_path)

        # Apply some transformation to the images.
        if self.transform:
            self.image = self.transform(self.image)        
        

        if self.train:
            if self.noise:
                if self.noise_type == 'uniform':
                    self.label_noise = self.lbl_noises[idx]
                    self.noise_variance = self.noise_variances[idx]
                    self.label = self.label + self.label_noise
                    # Apply normalization to the noisy training data.
                    if self.normalize:
                        self.image = normalize_images(self.image, self.images_mean, self.images_std)
                        self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                        # weight the noise value and its varince by the std of the labels of the dataset.
                        self.label_noise = self.label_noise/self.labels_std
                        self.noise_variance = self.noise_variance/self.labels_std
                    return (self.image, self.label, self.label_noise, self.noise_variance)
            else:
                # Apply normalization to the training data.
                if self.normalize:
                    self.image = normalize_images(self.image, self.images_mean, self.images_std)
                    self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
                return (self.image, self.label)
        else:
            # Apply normalization to the testing data.
            if self.normalize:
                self.image = normalize_images(self.image, self.images_mean, self.images_std)
                self.label = normalize_labels(self.label, self.labels_mean, self.labels_std)
            return (self.image, self.label)  

        

