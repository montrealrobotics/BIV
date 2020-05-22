from datetime import datetime

import pandas as pd
import torch
from torch.nn import MSELoss

from utils import group_testing, group_labels

import wandb


class Trainer:

    def __init__(self):

        """
        Description:
            A class with multiple training methods:
                - Baseline (The model without injecting any noises)
                - Baseline with noise.
                - Baseline with IV loss.
                - Baseline with batch normalized IV loss.

        Args:
            :cuda (bool) [private]: Controls if the model will be run on GPU or not.
        """

        # Check Cuda avaliability
        self.cuda = torch.cuda.is_available()

    def train(self, train_loader, test_loader, model,loss, optimizer,epochs):

        """
        Description:

            Train the network without injecting any noises.

        Return:
            Model
        Return type:
            nn.Module
        
        Args:
            :train_loader: A data loader for the training set.
            :test_loader:  A data loader for the testing set.
            :model: A random model. (i.e CNN).
            :loss: A loss function. (i.e MSE, IV, normalized IV)
            :optimizer: An optimizer. (i.e Adam)
            :epochs: Number of epochs
            """

        if self.cuda:
            ############################
            print("Running with Cuda support")
            ##############################

            train_runs_table = pd.DataFrame()
            test_runs_table = pd.DataFrame()
            for epoch in range(epochs):
                tr_losses = []
                tst_losses = []
                for i,(batch, labels) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    model.cuda(0)
                    batch = batch.cuda(0)
                    labels = torch.unsqueeze(labels,1).cuda(0)

                    out = model(batch)

                    mloss = loss(out,labels)
                
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


                # check test loss for categorical y_pred
                if epoch==9:
                    with torch.no_grad():
                        test_data = test_loader[0].cuda(0)
                        test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                        y_red_groups = group_testing(test_data, test_labels, 10, model, loss)


            
            time = str(datetime.now())
            train_runs_table.to_csv('/final_outps/train_run_at'+time+'.csv')
            test_runs_table.to_csv('/final_outps/test_run_at'+time+'.csv')

            #  Save the files in .csv format.
            wandb.save('/final_outps/train_run_at'+time+'.csv')
            wandb.save('/final_outps/test_run_at'+time+'.csv')


        else:
            ############################
            print("Running without Cuda support")
            ##############################

            train_runs_table = pd.DataFrame()
            test_runs_table = pd.DataFrame()
            for epoch in range(epochs):
                tr_losses = []
                tst_losses = []
                for i,(batch, labels) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    batch = batch
                    labels = torch.unsqueeze(labels,1)

                    out = model(batch)

                    mloss = loss(out,labels)
                
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

    def train_w_iv(self,train_loader, test_loader, model,loss, optimizer,epochs):

        """
        Description:

            Train the network with injecting noises.

        Return:
            Model
        Return type:
            nn.Module
        
        Args:
            :train_loader: A data loader for the training set.
            :test_loader:  A data loader for the testing set.
            :model: A random model. (i.e CNN).
            :loss: A loss function. (i.e MSE, IV, normalized IV)
            :optimizer: An optimizer. (i.e Adam)
            :epochs: Number of epochs
            """

        if self.cuda:
            #################################################
            print("Running using Cuda support")
            ##################################################

            model.cuda(0)   # Move the model to the GPU


            train_runs_table = pd.DataFrame()
            test_runs_table = pd.DataFrame()
            for epoch in range(epochs):
                
                tr_losses = []
                tst_losses = []
                for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
                    optimizer.zero_grad() 

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

                    # with torch.no_grad():
                    #     for i,(tst_batch, tst_labels,_,_) in enumerate(test_loader):

                    #     # test_data = test_loader[0].cuda(0)
                    #     # test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                    #         tst_batch = tst_batch.cuda(0)   # Move data to the GPU.
                    #         tst_labels = torch.unsqueeze(tst_labels,1).cuda(0)   # Move labels to the GPU.
                    #         out = model(tst_batch)
                    #         tloss = loss(out,tst_labels)
                    #         tst_losses.append(tloss.item())


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
            train_runs_table.to_csv('/final_outps/train_wiv_run_at'+time+'.csv')
            test_runs_table.to_csv('/final_outps/test_wiv_run_at'+time+'.csv')

            #  Save the files in .csv format.
            wandb.save('/final_outps/train_wiv_run_at'+time+'.csv')
            wandb.save('/final_outps/test_wiv_run_at'+time+'.csv')
        else:
            #################################################
            print("Running without Cuda support")
            ##################################################

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

                    # with torch.no_grad():
                    #     for i,(tst_batch, tst_labels,_,_) in enumerate(test_loader):
                    #     # test_data = test_loader[0]
                    #     # test_labels = torch.unsqueeze(test_loader[1],1)
                    #         tst_labels = torch.unsqueeze(tst_labels,1)
                    #         out = model(tst_batch)
                    #         tloss = loss(out,tst_labels)
                    #         tst_losses.append(tloss.item())

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
            train_runs_table.to_csv('./outputs/train_wiv_run_at'+time+'.csv')
            test_runs_table.to_csv('./outputs/test_wiv_run_at'+time+'.csv')

            #  Save the files in .csv format.
            wandb.save('./outputs/train_wiv_run_at'+time+'.csv')
            wandb.save('./outputs/test_wiv_run_at'+time+'.csv')

    def train_iv(self,train_loader, test_loader, model,loss, optimizer,epochs):

        """
        Description:

            Train the network using IV loss.

        Return:
            Model
        Return type:
            nn.Module
        
        Args:
            :train_loader: A data loader for the training set.
            :test_loader:  A data loader for the testing set.
            :model: A random model. (i.e CNN).
            :loss: IV (or normalized) loss function.
            :optimizer: An optimizer. (i.e Adam)
            :epochs: Number of epochs
            """


        if self.cuda:
            #################################################
            print("Running using Cuda support")
            ##################################################

            model.cuda(0)   # Move the model to the GPU
            mse_loss = MSELoss()    # Create mse loss for the testing part.
            

            train_runs_table = pd.DataFrame()
            test_runs_table = pd.DataFrame()
            for epoch in range(epochs):
                tr_losses = []
                tst_losses = []
                for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    batch = batch.cuda(0)
                    labels = torch.unsqueeze(labels,1).cuda(0)
                    lbls_noise = torch.unsqueeze(lbls_noise,1).type(torch.float32).cuda(0)
                    noises_var = torch.unsqueeze(noises_var,1).type(torch.float32).cuda(0)
                    labels_noisy = labels + lbls_noise
                    
    
                    out = model(batch)

                    mloss = loss(out,labels_noisy,noises_var)
                    # print(" Epoch: ",epoch," Batch Loss: ",mloss.item(), " Noises:  ",lbls_noise," Varinace of the noises: ", noises_var," IV: ", (1/noises_var))
                    # print("*"*20)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()

                    # with torch.no_grad():
                    #     for i,(tst_batch, tst_labels,_,_) in enumerate(test_loader):

                    #     # test_data = test_loader[0].cuda(0)
                    #     # test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                    #         tst_batch = tst_batch.cuda(0)   # Move data to the GPU.
                    #         tst_labels = torch.unsqueeze(tst_labels,1).cuda(0)   # Move labels to the GPU.
                    #         out = model(tst_batch)
                    #         tloss = mse_loss(out,tst_labels)
                    #         tst_losses.append(tloss.item())

                    with torch.no_grad():

                        test_data = test_loader[0].cuda(0)
                        test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                        out = model(test_data)
                        tloss = mse_loss(out,test_labels)
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
            train_runs_table.to_csv('/final_outps/train_iv_run_at'+time+'.csv')
            test_runs_table.to_csv('/final_outps/test_iv_run_at'+time+'.csv')

            #  Save the files in .csv format.
            wandb.save('/final_outps/train_iv_run_at'+time+'.csv')
            wandb.save('/final_outps/test_iv_run_at'+time+'.csv')
        else:
            #################################################
            print("Running without Cuda support")
            ##################################################
            # Create mse loss for the testing part.
            mse_loss = MSELoss()

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

                    mloss = loss(out,labels_noisy,noises_var)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()

                    # with torch.no_grad():
                    #     for i,(tst_batch, tst_labels,_,_) in enumerate(test_loader):
                    #     # test_data = test_loader[0]
                    #     # test_labels = torch.unsqueeze(test_loader[1],1)
                    #         tst_labels = torch.unsqueeze(tst_labels,1)
                    #         out = model(tst_batch)
                    #         tloss = mse_loss(out,tst_labels)
                    #         tst_losses.append(tloss.item())


                    with torch.no_grad():
                        test_data = test_loader[0]
                        test_labels = torch.unsqueeze(test_loader[1],1)
    
                        out = model(test_data)
                        tloss = mse_loss(out,test_labels)
                        tst_losses.append(tloss.item())


                # save train losses for each epoch in the runs table for future calculations.
                train_runs_table = pd.concat([train_runs_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save train losses for each epoch in the runs table for future calculations.
                test_runs_table = pd.concat([test_runs_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.")
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})
            
            time = str(datetime.now())
            train_runs_table.to_csv('./outputs/train_iv_run_at'+time+'.csv')
            test_runs_table.to_csv('./outputs/test_iv_run_at'+time+'.csv')

            #  Save the files in .csv format.
            wandb.save('./outputs/train_iv_run_at'+time+'.csv')
            wandb.save('./outputs/test_iv_run_at'+time+'.csv')
