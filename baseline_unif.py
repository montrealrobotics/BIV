###################################################################################################

# Things to do before ruuning the code on the server:

# Change the dataset directory on the dataloder from: "./Dataset/UTKface/*" to "/datasets/UTKface/*"
# Change Wandb log file from: test      to: estimation_baseline_with_noise
###################################################################################################

import glob
import os
import math
import argparse

from datetime import datetime
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch import nn
from torch import optim


import wandb


# Configure global settings

torch.manual_seed(42)
wandb.init(project="test", entity="khamiesw")

# Get the api key from the environment variables.
api_key = os.environ.get('WANDB_API_KEY')
print("API Key:",api_key)
# login to my wandb account.
wandb.login(api_key)


class UTKface(Dataset):

    def __init__(self, path, train = True, transform = None, noise = False, noise_type = None, uniform_data = None):
        self.train = train
        self.images_path = glob.glob(path)
        self.transform = transform
        self.noise = noise
        self.noise_type = noise_type
        self.unif_data = uniform_data
        
    
    def __spilit_data(self):
        train_labels = []
        if self.train:
            train_img_paths = self.images_path[:16000]
            for path in train_img_paths:
                label = path.split("/")[3].split("_")[0]
                train_labels.append(int(label))
            return (train_img_paths,train_labels)

        else:
            test_labels = []
            test_img_paths = self.images_path[16000:]
            for path in test_img_paths:
                label = path.split("/")[3].split("_")[0]
                test_labels.append(int(label))
            return (test_img_paths,test_labels)

    def gauss_label_noise(self):
        global_mean = 0
        variance_dist = torch.distributions.normal.Normal(global_mean,self.v_variance)

        noise_variance = variance_dist.sample((1,))
        noise_variance = torch.abs(noise_variance).item()
        
        noisy_label = torch.distributions.normal.Normal(global_mean, noise_variance).sample((1,)).item() 
        return (noisy_label, noise_variance)
    
    def get_unif_limits(self, mu,v):
        b = mu + math.sqrt(3*v)
        a = mu - math.sqrt(3*v)

        return a,b
    
    def unif_label_noise(self, a,b):

        variance_dist = torch.distributions.uniform.Uniform(a,b)
        v_variance = variance_dist.variance
        mean_variance = variance_dist.mean
            
        # Sample noise variance.
        noise_variance = variance_dist.sample((1,)).item()

            
        noisy_label = torch.distributions.normal.Normal(0, noise_variance).sample((1,)).item() 
        return (noisy_label, noise_variance)


    def __len__(self):
        return len(self.__spilit_data()[0])

    def __getitem__(self, idx):
        images_pth, labels = self.__spilit_data()
        img_path = images_pth[idx]
        label = labels[idx]

        # Convert the label to a tensor

        label = torch.tensor(label,dtype=torch.float32)

        image = Image.open(img_path)

        if self.noise:
            if self.noise_type == 'uniform':
                a,b = self.get_unif_limits(self.unif_data[0], self.unif_data[1])
                lbl_noise, noise_variance = self.unif_label_noise(a, b)
            elif self.noise_type == 'gauss':
                lbl_noise, noise_variance = self.gauss_label_noise()
            else:
                print("Exception: you must specify a noise, either 'uniform' or 'gauss' ")
                return 0
        else:
            # Set the noise and its variance to a default value: 0
            lbl_noise = 0
            noise_variance = 1e-10   # samll number, for numerical stability.
        
        if self.transform:
            image = self.transform(image)

        return (image, label,lbl_noise,noise_variance)

# Model

class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv1.apply(self.custom_init)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels = 20, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels = 32, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels = 64, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels = 128, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels = 256, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.linear = nn.Linear(in_features = 256*3*3 , out_features = 1 )
        # self.linear.apply(self.custom_init)

    def custom_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0, std=100)
            # m.bias.data.fill_(0)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(out.size()[0],-1)
        out = self.linear(out)

        return out


def train_cuda(train_loader, test_loader, model,loss, optimizer,epochs):
    
    train_runs_table = pd.DataFrame()
    test_runs_table = pd.DataFrame()
    for epoch in range(epochs):
        tr_losses = []
        tst_losses = []
        for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
            optimizer.zero_grad() 

            model.cuda(0)
            batch = batch.cuda(0)
            labels = torch.unsqueeze(labels,1).cuda(0)
            lbls_noise = torch.unsqueeze(lbls_noise,1).type(torch.float32).cuda(0)
            noises_var = torch.unsqueeze(noises_var,1).type(torch.float32).cuda(0)
            labels_noisy = labels + lbls_noise


            out = model(batch)

            mloss = loss(out,labels_noisy)
        
            tr_losses.append(mloss.item())
            mloss.backward()
       
            optimizer.step()

            with torch.no_grad():
                test_data = test_loader[0].cuda(0)
                test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)
                out = model(test_data)
                tloss = loss(out,test_labels)
                tst_losses.append(tloss.item())

        # save the train losses in the runs table future calculation.
        train_runs_table = pd.concat([train_runs_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
        
        # save the train losses in the runs table future calculation.
        test_runs_table = pd.concat([test_runs_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

        print('Epoch:', epoch, "has finished.")
        # log the results to wandb 
        for i in range(len(tr_losses)):
             wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})
    
    time = str(datetime.now())
    train_runs_table.to_csv('/final_outps/train_run_at'+time+'.csv')
    test_runs_table.to_csv('/final_outps/test_run_at'+time+'.csv')

    #  Save the files in .csv format.
    wandb.save('/final_outps/train_run_at'+time+'.csv')
    wandb.save('/final_outps/test_run_at'+time+'.csv')

def train_cudaless(train_loader, test_loader, model,loss, optimizer,epochs):
    
    train_runs_table = pd.DataFrame()
    test_runs_table = pd.DataFrame()
    for epoch in range(epochs):
        tr_losses = []
        tst_losses = []
        for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
            optimizer.zero_grad() 

            batch = batch
            labels = torch.unsqueeze(labels,1)
            lbls_noise = torch.unsqueeze(lbls_noise,1).type(torch.float32)
            noises_var = torch.unsqueeze(noises_var,1).type(torch.float32)
            labels_noisy = labels + lbls_noise


            out = model(batch)

            mloss = loss(out,labels_noisy)
        
            tr_losses.append(mloss.item())
            mloss.backward()
       
            optimizer.step()

            with torch.no_grad():
                test_data = test_loader[0]
                test_labels = torch.unsqueeze(test_loader[1],1)
                out = model(test_data)
                tloss = loss(out,test_labels)
                tst_losses.append(tloss.item())

        # save the train losses in the runs table future calculation.
        train_runs_table = pd.concat([train_runs_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
        
        # save the train losses in the runs table future calculation.
        test_runs_table = pd.concat([test_runs_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

        print('Epoch:', epoch, "has finished.")
        # log the results to wandb 
        for i in range(len(tr_losses)):
             wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})
    
    time = str(datetime.now())
    train_runs_table.to_csv('./outputs/train_run_at'+time+'.csv')
    test_runs_table.to_csv('./outputs/test_run_at'+time+'.csv')

    #  Save the files in .csv format.
    wandb.save('./outputs/train_run_at'+time+'.csv')
    wandb.save('./outputs/test_run_at'+time+'.csv')

# Main 

if __name__ == "__main__":

    # Parse arguments from the commandline
    
    parser =  argparse.ArgumentParser(description=" A parser for baseline uniform noisy experiment")

    parser.add_argument("--mu" , type=int, default=0)
    parser.add_argument("--v", type= int, default=1)

    args = parser.parse_args()
    unif_data = (args.mu,args.v) 

    trans= torchvision.transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    # uniform_data = [(2,1),(5,1),(5,4),\
    #                     (10,1),(10,4),(10,10),\
    #                     (20,1),(20,4),(20,10),(20,50),\
    #                     (50,1),(50,4),(50,10),(50,50),(50,150),\
    #                     (150,1),(150,4),(150,10),(150,50),(150,150),(150,1000)]

                             # Take the first sample.

    train_data = UTKface("./Datasets/UTKFace/*", transform= trans, train= True, noise= True, noise_type='uniform', uniform_data = unif_data) 
    test_data = UTKface("./Datasets/UTKFace/*", transform= trans, train= False, noise= False)

    
    train_loader = DataLoader(train_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=1000)


    model = AgeModel()
    loss = torch.nn.MSELoss()
    optimz = optim.Adam(model.parameters(), lr=3e-4)
    epochs = 20

    train_dataset = train_loader
    test_dataset = iter(test_loader).next()

    # wandb.watch(model)

    # Check Cuda avaliability
    if torch.cuda.is_available():
        print("Running using Cuda support")
        train_cuda(train_dataset,test_dataset,model,loss,optimz,epochs)
    else:
        print("Running without Cuda support")
        train_cudaless(train_dataset,test_dataset,model,loss,optimz,epochs)
