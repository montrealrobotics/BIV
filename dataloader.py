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
from utils import flip_coin


class UTKface(Dataset):

    def __init__(self, path, train = True, transform = None, noise = False , noise_type = None, \
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
        self.transform = transform
        self.noise = noise
        self.noise_type = noise_type
        self.dist_data = distribution_data
        self.train_size= d_params.get('train_size')
        self.noise_threshold = noise_threshold
        self.threshold_value = threshold_value
        
        # This is not the proper implementation, but doing that for research purproses.

        # Load the dataset
        self.images_pth, self.labels = self.load_data()
        # Load the normalization constant variables.
        self.images_mean, self.images_std, self.labels_mean, self.labels_std = get_dataset_stats()
        # Generate noise for the training samples.
        if self.train:
            if self.noise:
                if self.noise_type == 'uniform':
                    self.lbl_noises, self.noise_variances = self.generate_noise(norm = self.normalize)  # normalize the noise.  
                else:
                    raise ValueError("Gamma noise is not supported at the current moment.")

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

                filt_images_path.append(self.images_pth[idx])
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
        else:
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
            a,b = self.get_uniform_params(mu, v)
            var_dist = torch.distributions.uniform.Uniform(a,b)
        
        elif dist_type == "gamma":
            alpha, beta = self.get_gamma_params(mu,v)
            var_dist = torch.distributions.gamma.Gamma(alpha,beta)

        return var_dist


    def gaussian_noise(self, var_dists, noise_complexity="simple"):

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
        

        if noise_complexity == "simple":
            noises_vars = var_dists.get("1").sample((self.train_size,))
            noises = [torch.distributions.normal.Normal(0, torch.sqrt(var)).sample((1,)).item() for var in noises_vars]
            return (noises, noises_vars)

        else:
            low = 0
            high = 0
            noises_vars = []
            for idx in range(self.train_size):
                coin_decision = flip_coin(c_type=self.coin_fairness)
                if coin_decision == "low":
                    noises_vars.append(var_dists.get("1").sample((1,)))
                    low+=1
                else:
                    noises_vars.append(var_dists.get("2").sample((1,)))
                    high+=1

            print("Sample ratio from the low noise distribution: ", low/self.train_size*100)
            print("Sample ratio from the high noise distribution: ",high/self.train_size*100)

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
        if self.noise_type == "uniform":
            noise_complexity = self.dist_data.get('noise_complexity')
            print("Uniform is: {}".format(noise_complexity))

            mu = self.dist_data.get("mu")
            is_vmax = self.dist_data.get("is_vmax") # True or False
            if is_vmax:
                v = get_unif_Vmax(mu, scale_value=self.dist_data.get("vmax_scale"))
            else:
                v =  self.dist_data["v"] 


            if noise_complexity == "simple":
                dists["1"] = self.get_distribution(self.noise_type,mu,v)
                lbl_noises, noise_variances = self.gaussian_noise(dists)
                return (lbl_noises, noise_variances)

            else:
                mu_2 = self.dist_data.get("mu_unf_2")
                is_vmax_2 = self.dist_data.get("is_vmax_unf_2") # True or False
                if is_vmax_2:
                    v_2 = get_unif_Vmax(mu_2, scale_value=self.dist_data.get("vmax_scale_unf_2"))
                else:
                    v_2 =  self.dist_data["v_unf_2"] 
                # flip a coin for choosing between the two distributions.
                data = [(mu,v),(mu_2,v_2)]
                for idx, d in enumerate(data):
                    dists[str(idx+1)] = self.get_distribution(self.noise_type,d[0],d[1])

                # set the fairness of the coin.
                self.coin_fairness = self.dist_data.get("coin_fairness")
            
        
                lbl_noises, noise_variances = self.gaussian_noise(dists, noise_complexity)
                return (lbl_noises, noise_variances)
            

        elif self.noise_type == "gamma":
            mu = self.dist_data.get("mu")
            v =  self.dist_data.get("v")
            dists["1"] = self.get_distribution(self.noise_type,mu,v)
            lbl_noises, noise_variances = self.gaussian_noise(dists)
            return (lbl_noises, noise_variances)
        
        else:
            raise ValueError("{} is not supported yet.".format(self.noise_type))


            
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
                        self.noise_variance = self.noise_variance/(self.labels_std**2)
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

        

