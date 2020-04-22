import glob
import os
import math

from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset


class UTKface(Dataset):

    def __init__(self, path, train = True, transform = None, noise_type = None, uniform_data = None):

        """
        Description:
            UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). 
            The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. 
            The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. 


        Args:
            :train (bool): Controls if the data will include the noise and varaince with the actual inputs and labels or not.
            :train_size (int): Controls the number of samples in the training set.
            :images_path (string): Dataset directory path.
            :transform (Object): A torchvision transofrmer for processing the images.
            :noise_type (string): Controls the type of noise that will be added to the data
            :unif_data (tuple): A tuple of the mean and the variance of the uniform distribution noise.
        
        """
        self.train = train
        self.data_paths = glob.glob(path)
        self.dataset_size  = len(self.data_paths)
        self.transform = transform
        self.noise_type = noise_type
        self.unif_data = uniform_data
        self.train_size=16000
        

        # This is not the proper implementation, but doing that for research purproses.

        # Load the dataset
        self.images_pth, self.labels = self.__load_data()
        

        # Generate noise for the training samples.
        if self.train:
            if self.noise_type == 'uniform':
                self.lbl_noises, self.noise_variances = self.generate_noise()       
            # elif self.noise_type == 'gauss':
            #     lbl_noise, noise_variance = self.gauss_label_noise()
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
            return (train_img_paths,train_labels)

        else:
            test_labels = []
            test_img_paths = self.data_paths[self.train_size:]
            for path in test_img_paths:
                label = path.split("/")[3].split("_")[0]
                test_labels.append(int(label))
            return (test_img_paths,test_labels)

    # def gauss_label_noise(self):
    #     global_mean = 0
    #     variance_dist = torch.distributions.normal.Normal(global_mean,self.v_variance)

    #     noise_variance = variance_dist.sample((1,))
    #     noise_variance = torch.abs(noise_variance).item()
        
    #     noisy_label = torch.distributions.normal.Normal(global_mean, noise_variance).sample((1,)).item() 
    #     return (noisy_label, noise_variance)
    
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
    

    def gaussian_noise(self, a,b):

        """ Description:
            Generates gaussian noises with mean 0 and heteroscedasticitical variance that sampled from a unifrom distribution.

        Return:
            Guassian noises and their heteroscedasticitical variances.


        Return type:
            Tuple.

        Args:
            :a (float): Uniform lower bound.
            :b (float): Uniform upper bound.
        """
        variance_dist = torch.distributions.uniform.Uniform(a,b)
            
        # Sample heteroscedasticitical noises variances for the whole training set.
        noise_variances = variance_dist.sample((self.train_size,))

        # Sample gaussian noises.
        noisy_labels = [torch.distributions.normal.Normal(0, var).sample((1,)).item() for var in noise_variances]
        return (noisy_labels, noise_variances)

    
    def generate_noise(self):

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
        
        a,b = self.get_unif_bounds(self.unif_data[0], self.unif_data[1])
        lbl_noises, noise_variances = self.gaussian_noise(a, b)
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
        return self.dataset_size

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

        if self.train:
            img_path = self.images_pth[idx]
            label = self.labels[idx]

        if self.noise_type == 'uniform'
                noise = self.lbl_noises[idx]
                variance = self.noise_variances[idx]

            # Convert the label to a tensor

            label = torch.tensor(label,dtype=torch.float32)
            image = Image.open(img_path)


            if self.transform:
                image = self.transform(image)

            return (image, label,noise,variance)
        else:
            img_path = self.images_pth[idx]
            label = self.labels[idx]

            # Convert the label to a tensor

            label = torch.tensor(label,dtype=torch.float32)
            image = Image.open(img_path)


            if self.transform:
                image = self.transform(image)

            return (image, label)  
