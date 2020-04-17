import glob
import os
import math

from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset


class UTKface(Dataset):

    def __init__(self, path, train = True, transform = None, noise_type = None, uniform_data = None):
        self.train = train
        self.images_path = glob.glob(path)
        self.transform = transform
        self.noise_type = noise_type
        self.unif_data = uniform_data
        self.train_size=16000

        # This is not the proper implementation, but doing that for research purproses.

        self.images_pth, self.labels = self.__spilit_data()         # Load the dataset

        if self.train:
            if self.noise_type == 'uniform':
                self.lbl_noises, self.noise_variances = self.generate_noise()       # Generate noises for training data.
            # elif self.noise_type == 'gauss':
            #     lbl_noise, noise_variance = self.gauss_label_noise()
            else:
                print("Exception: you must specify a noise, either 'uniform' or 'gauss' ")
                return 0
        
    
    def __spilit_data(self):
        train_labels = []
        if self.train:
            train_img_paths = self.images_path[:self.train_size]
            for path in train_img_paths:
                label = path.split("/")[3].split("_")[0]
                train_labels.append(int(label))
            return (train_img_paths,train_labels)

        else:
            test_labels = []
            test_img_paths = self.images_path[self.train_size:]
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
    
    def get_unif_limits(self, mu,v):
        b = mu + math.sqrt(3*v)
        a = mu - math.sqrt(3*v)

        return a,b
    
    def unif_noise(self, a,b):

        variance_dist = torch.distributions.uniform.Uniform(a,b)
        v_variance = variance_dist.variance
        mean_variance = variance_dist.mean
            
        # Sample noise variance.
        noise_variance = variance_dist.sample((1,)).item()

            
        noisy_label = torch.distributions.normal.Normal(0, noise_variance).sample((1,)).item() 
        return (noisy_label, noise_variance)

    def unif_noise_batch(self, a,b):

        variance_dist = torch.distributions.uniform.Uniform(a,b)
            
        # Sample noise variance for the whole dataset.
        noise_variances = variance_dist.sample((self.train_size,))

            
        # noisy_labels = torch.distributions.normal.Normal(0, noise_variances).sample((self.train_size,))
        noisy_labels = [torch.distributions.normal.Normal(0, var).sample((1,)).item() for var in noise_variances]
        return (noisy_labels, noise_variances)

    
    def generate_noise(self):
        a,b = self.get_unif_limits(self.unif_data[0], self.unif_data[1])
        lbl_noises, noise_variances = self.unif_noise_batch(a, b)
        return (lbl_noises, noise_variances)

            
    def __len__(self):
        return len(self.__spilit_data()[0])

    def __getitem__(self, idx):

        if self.train:
            img_path = self.images_pth[idx]
            label = self.labels[idx]

        
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









































# import glob
# import os
# import math

# from PIL import Image

# import torch
# import torchvision
# from torch.utils.data import Dataset


# class UTKface(Dataset):

#     def __init__(self, path, train = True, transform = None, noise = False, noise_type = None, uniform_data = None):
#         self.train = train
#         self.images_path = glob.glob(path)
#         self.transform = transform
#         self.noise = noise
#         self.noise_type = noise_type
#         self.unif_data = uniform_data
        
#         self.train_size = 16000         # Set train size.
#         # self.reset_counter = 0          # A counter for reseting the noise over the epochs to be the same to each labels.
        
    
#     def __spilit_data(self):
#         train_labels = []
#         if self.train:
#             train_img_paths = self.images_path[:self.train_size]
#             for path in train_img_paths:
#                 label = path.split("/")[3].split("_")[0]
#                 train_labels.append(int(label))
#             return (train_img_paths,train_labels)

#         else:
#             test_labels = []
#             test_img_paths = self.images_path[self.train_size:]
#             for path in test_img_paths:
#                 label = path.split("/")[3].split("_")[0]
#                 test_labels.append(int(label))
#             return (test_img_paths,test_labels)

#     def gauss_label_noise(self):
#         global_mean = 0
#         variance_dist = torch.distributions.normal.Normal(global_mean,self.v_variance)

#         noise_variance = variance_dist.sample((1,))
#         noise_variance = torch.abs(noise_variance).item()
        
#         noisy_label = torch.distributions.normal.Normal(global_mean, noise_variance).sample((1,)).item() 
#         return (noisy_label, noise_variance)
    
#     def get_unif_limits(self, mu,v):
#         b = mu + math.sqrt(3*v)
#         a = mu - math.sqrt(3*v)

#         return a,b
    
#     def unif_label_noise(self, a,b):

#         variance_dist = torch.distributions.uniform.Uniform(a,b)
#         v_variance = variance_dist.variance
#         mean_variance = variance_dist.mean
            
#         # Sample noise variance.
#         noise_variance = variance_dist.sample((1,)).item()

            
#         noisy_label = torch.distributions.normal.Normal(0, noise_variance).sample((1,)).item() 
#         return (noisy_label, noise_variance)


#     def __len__(self):
#         return len(self.__spilit_data()[0])

#     def __getitem__(self, idx):
#         images_pth, labels = self.__spilit_data()
#         img_path = images_pth[idx]
#         label = labels[idx]

#         # Convert the label to a tensor

#         label = torch.tensor(label,dtype=torch.float32)

#         image = Image.open(img_path)

#         if self.noise:
#             if self.noise_type == 'uniform':
#                 a,b = self.get_unif_limits(self.unif_data[0], self.unif_data[1])
#                 lbl_noise, noise_variance = self.unif_label_noise(a, b)
#             elif self.noise_type == 'gauss':
#                 lbl_noise, noise_variance = self.gauss_label_noise()
#             else:
#                 print("Exception: you must specify a noise, either 'uniform' or 'gauss' ")
#                 return 0
#         else:
#             # Set the noise and its variance to a default value: 0
#             lbl_noise = 0
#             noise_variance = 1e-10   # samll number, for numerical stability.
        
#         if self.transform:
#             image = self.transform(image)
        


#         return (image, label,lbl_noise,noise_variance)
