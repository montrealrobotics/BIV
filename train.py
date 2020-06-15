import itertools
from datetime import datetime

import pandas as pd

import torch
from torch.nn import MSELoss

import matplotlib.pyplot as plt

from utils import group_testing, group_labels, get_dataset_stats, normalize_images, normalize_labels, plot_hist

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

            train_loss_table = pd.DataFrame()
            test_loss_table = pd.DataFrame()

            # Dataframes for saving the training and testing labels into .csv files.
            train_labels_table = pd.DataFrame()
            test_labels_table = pd.DataFrame()

            # Dataframes for saving the training and testing predictions into .csv files.
            train_out_table = pd.DataFrame()
            test_out_table = pd.DataFrame()

            



            for epoch in range(epochs):
                #Saving the train and test losses for logging and visualization purposes.
                tr_losses = []
                tst_losses = []

                # Saving the train and test predictions for logging and visualization purposes.
                tr_out = []
                tst_out = []

                # Saving the train and test labels for logging and visualization purposes.
                tr_labels = []
                tst_labels = []
               
                for i,(batch, labels) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    # Feed the data to the model
                    model.cuda(0)
                    batch = batch.cuda(0)
                    labels = torch.unsqueeze(labels,1).cuda(0)
                    

                    out = model(batch)  

                    mloss = loss(out,labels)
                    tr_losses.append(mloss.item())

                    # Update the parameters
                    mloss.backward()
            
                    optimizer.step()


                    if epoch == (epochs-1):
                         # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                        tr_out.append(out.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_out_list = list(itertools.chain.from_iterable(tr_out))
                        train_out_table =  pd.DataFrame(tr_out_list, columns = [str(epoch)])

                        train_out_table.to_csv('/final_outps/train_out.csv') 
                        wandb.save('/final_outps/train_out.csv')

                         # 2) Convert the train labels to a dataframe. (y)
                        tr_labels.append(labels.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_labels_list = list(itertools.chain.from_iterable(tr_labels))
                        train_labels_table =  pd.DataFrame(tr_labels_list, columns = [str(epoch)])


                        # Save the train labels to csv file ,then upload it to wandb.
                        train_labels_table.to_csv('/final_outps/train_labels.csv')
                        wandb.save('/final_outps/train_labels.csv')

                    with torch.no_grad():
                        test_data = test_loader[0].cuda(0)
                        test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)
                        out = model(test_data)
                        tloss = loss(out,test_labels)
                        tst_losses.append(tloss.item()) 
                    
                        if epoch == (epochs-1):

                            # 1) Convert predictions of the test labels in the last epoch to a dataframe. (y_)
                            tst_out.append(out.view(1,-1).squeeze(0).tolist()) 
        
                            # unpack all tr_out from list of lists to a one list.
                            tst_out_list = list(itertools.chain.from_iterable(tst_out))
                            test_out_table =  pd.DataFrame(tst_out_list, columns = [str(epoch)])

                            # Save to csv and upload to wandb.
                            test_out_table.to_csv('/final_outps/test_out.csv')
                            wandb.save('/final_outps/test_out.csv')

                            # 2) Convert test labels to a dataframe. (y)
                            test_labels_flatt = test_labels.view(1,-1).squeeze(0).tolist()
                            test_labels_table =  pd.DataFrame(test_labels_flatt, columns = [str(epoch)])

                            test_labels_table.to_csv('/final_outps/test_labels.csv')
                            wandb.save('/final_outps/test_labels.csv')

                            # Plot the histogram for the testing predictions

                            # plot_hist(tst_out_list,"y_test_")

                            # plot_hist(test_labels_flatt,"y_test")
                   
                            

                # save the train losses in the runs table future calculation.
                train_loss_table = pd.concat([train_loss_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save the train losses in the runs table future calculation.
                test_loss_table = pd.concat([test_loss_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.") 
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})


                # # check test loss for categorical y_pred
                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0].cuda(0)
                #         test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss)

            # Save the train and test losses to .csv files and upload it to the server.

            time = str(datetime.now())
            train_loss_table.to_csv('/final_outps/train_loss_at'+time+'.csv')
            test_loss_table.to_csv('/final_outps/test_loss_at'+time+'.csv')
            wandb.save('/final_outps/train_loss_at'+time+'.csv')
            wandb.save('/final_outps/test_loss_at'+time+'.csv')


        else:
            ############################
            print("Running without Cuda support")
            ##############################

            train_loss_table = pd.DataFrame()
            test_loss_table = pd.DataFrame()

            # Dataframes for saving the training and testing labels into .csv files.
            train_labels_table = pd.DataFrame()
            test_labels_table = pd.DataFrame()

            # Dataframes for saving the training and testing predictions into .csv files.
            train_out_table = pd.DataFrame()
            test_out_table = pd.DataFrame()



            for epoch in range(epochs):
                #Saving the train and test losses for logging and visualization purposes.
                tr_losses = []
                tst_losses = []

                # Saving the train and test predictions for logging and visualization purposes.
                tr_out = []
                tst_out = []

                # Saving the train and test labels for logging and visualization purposes.
                tr_labels = []
                tst_labels = []


                for i,(batch, labels) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    labels = torch.unsqueeze(labels,1)

                    out = model(batch)

                    mloss = loss(out,labels)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()


                    if epoch == (epochs-1):
                         # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                        tr_out.append(out.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_out_list = list(itertools.chain.from_iterable(tr_out))
                        train_out_table =  pd.DataFrame(tr_out_list, columns = [str(epoch)])

                        train_out_table.to_csv('./outputs/train_out.csv') 
                        wandb.save('./outputs/train_out.csv')

                         # 2) Convert the train labels to a dataframe. (y)
                        tr_labels.append(labels.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_labels_list = list(itertools.chain.from_iterable(tr_labels))
                        train_labels_table =  pd.DataFrame(tr_labels_list, columns = [str(epoch)])


                        # Save the train labels to csv file ,then upload it to wandb.
                        train_labels_table.to_csv('./outputs/train_labels.csv')
                        wandb.save('./outputs/train_labels.csv')
                        

                    with torch.no_grad():
                        test_data = test_loader[0]
                        test_labels = torch.unsqueeze(test_loader[1],1)
                        out = model(test_data)
                        tloss = loss(out,test_labels)
                        tst_losses.append(tloss.item())


                        if epoch == (epochs-1):

                            # 1) Convert predictions of the test labels in the last epoch to a dataframe. (y_)
                            tst_out.append(out.view(1,-1).squeeze(0).tolist()) 
        
                            # unpack all tr_out from list of lists to a one list.
                            tst_out_list = list(itertools.chain.from_iterable(tst_out))
                            test_out_table =  pd.DataFrame(tst_out_list, columns = [str(epoch)])

                            # Save to csv and upload to wandb.
                            test_out_table.to_csv('./outputs/test_out.csv')
                            wandb.save('./outputs/test_out.csv')

                            # 2) Convert test labels to a dataframe. (y)
                            test_labels_flatt = test_labels.view(1,-1).squeeze(0).tolist()
                            test_labels_table =  pd.DataFrame(test_labels_flatt, columns = [str(epoch)])

                            test_labels_table.to_csv('./outputs/test_labels.csv')
                            wandb.save('./outputs/test_labels.csv')


                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0]
                #         test_labels = torch.unsqueeze(test_loader[1],1)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss) 




                # save the train losses in the runs table future calculation.
                train_loss_table = pd.concat([train_loss_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save the train losses in the runs table future calculation.
                test_loss_table = pd.concat([test_loss_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.") 
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})


                # # check test loss for categorical y_pred
                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0].cuda(0)
                #         test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss)

            # Save the train and test losses to .csv files and upload it to the server.

            time = str(datetime.now())
            train_loss_table.to_csv('./outputs/train_loss_at'+time+'.csv')
            test_loss_table.to_csv('./outputs/test_loss_at'+time+'.csv')
            wandb.save('./outputs/train_loss_at'+time+'.csv')
            wandb.save('./outputs/test_loss_at'+time+'.csv')

    def train_w_iv(self,train_loader, test_loader, model,loss, optimizer, epochs):

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

            train_loss_table = pd.DataFrame()
            test_loss_table = pd.DataFrame()

            # Dataframes for saving the training and testing labels into .csv files.
            train_labels_table = pd.DataFrame()
            test_labels_table = pd.DataFrame()

            # Dataframes for saving the training and testing predictions into .csv files.
            train_out_table = pd.DataFrame()
            test_out_table = pd.DataFrame()

            
            for epoch in range(epochs):
                
                #Saving the train and test losses for logging and visualization purposes.
                tr_losses = []
                tst_losses = []

                # Saving the train and test predictions for logging and visualization purposes.
                tr_out = []
                tst_out = []

                # Saving the train and test labels for logging and visualization purposes.
                tr_labels = []
                tst_labels = []
               
                for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    batch = batch.cuda(0)
                    labels = torch.unsqueeze(labels,1).cuda(0)

                    out = model(batch)

                    mloss = loss(out,labels)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()

                    if epoch == (epochs-1):
                         # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                        tr_out.append(out.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_out_list = list(itertools.chain.from_iterable(tr_out))
                        train_out_table =  pd.DataFrame(tr_out_list, columns = [str(epoch)])

                        train_out_table.to_csv('/final_outps/train_out.csv') 
                        wandb.save('/final_outps/train_out.csv')

                         # 2) Convert the train labels to a dataframe. (y)
                        tr_labels.append(labels.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_labels_list = list(itertools.chain.from_iterable(tr_labels))
                        train_labels_table =  pd.DataFrame(tr_labels_list, columns = [str(epoch)])


                        # Save the train labels to csv file ,then upload it to wandb.
                        train_labels_table.to_csv('/final_outps/train_labels.csv')
                        wandb.save('/final_outps/train_labels.csv')


                    with torch.no_grad():

                        test_data = test_loader[0].cuda(0)
                        test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                        out = model(test_data)
                        tloss = loss(out,test_labels)
                        tst_losses.append(tloss.item())

                        if epoch == (epochs-1):

                            # 1) Convert predictions of the test labels in the last epoch to a dataframe. (y_)
                            tst_out.append(out.view(1,-1).squeeze(0).tolist()) 
        
                            # unpack all tr_out from list of lists to a one list.
                            tst_out_list = list(itertools.chain.from_iterable(tst_out))
                            test_out_table =  pd.DataFrame(tst_out_list, columns = [str(epoch)])

                            # Save to csv and upload to wandb.
                            test_out_table.to_csv('/final_outps/test_out.csv')
                            wandb.save('/final_outps/test_out.csv')

                            # 2) Convert test labels to a dataframe. (y)
                            test_labels_flatt = test_labels.view(1,-1).squeeze(0).tolist()
                            test_labels_table =  pd.DataFrame(test_labels_flatt, columns = [str(epoch)])

                            test_labels_table.to_csv('/final_outps/test_labels.csv')
                            wandb.save('/final_outps/test_labels.csv')



                # save the train losses in the runs table future calculation.
                train_loss_table = pd.concat([train_loss_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save the train losses in the runs table future calculation.
                test_loss_table = pd.concat([test_loss_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.") 
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})


                # # check test loss for categorical y_pred
                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0].cuda(0)
                #         test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss)

            # Save the train and test losses to .csv files and upload it to the server.

            time = str(datetime.now())
            train_loss_table.to_csv('/final_outps/train_loss_at'+time+'.csv')
            test_loss_table.to_csv('/final_outps/test_loss_at'+time+'.csv')
            wandb.save('/final_outps/train_loss_at'+time+'.csv')
            wandb.save('/final_outps/test_loss_at'+time+'.csv')
            
        else:
            #################################################
            print("Running without Cuda support")
            ##################################################

            train_loss_table = pd.DataFrame()
            test_loss_table = pd.DataFrame()

            # Dataframes for saving the training and testing labels into .csv files.
            train_labels_table = pd.DataFrame()
            test_labels_table = pd.DataFrame()

            # Dataframes for saving the training and testing predictions into .csv files.
            train_out_table = pd.DataFrame()
            test_out_table = pd.DataFrame()

    
            for epoch in range(epochs):
                
                #Saving the train and test losses for logging and visualization purposes.
                tr_losses = []
                tst_losses = []

                # Saving the train and test predictions for logging and visualization purposes.
                tr_out = []
                tst_out = []

                # Saving the train and test labels for logging and visualization purposes.
                tr_labels = []
                tst_labels = []

                for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    batch = batch
                    labels = torch.unsqueeze(labels,1)
                    out = model(batch)
                    mloss = loss(out,labels)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()

                    if epoch == (epochs-1):
                         # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                        tr_out.append(out.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_out_list = list(itertools.chain.from_iterable(tr_out))
                        train_out_table =  pd.DataFrame(tr_out_list, columns = [str(epoch)])

                        train_out_table.to_csv('./outputs/train_out.csv') 
                        wandb.save('./outputs/train_out.csv')

                         # 2) Convert the train labels to a dataframe. (y)
                        tr_labels.append(labels.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_labels_list = list(itertools.chain.from_iterable(tr_labels))
                        train_labels_table =  pd.DataFrame(tr_labels_list, columns = [str(epoch)])


                        # Save the train labels to csv file ,then upload it to wandb.
                        train_labels_table.to_csv('./outputs/train_labels.csv')
                        wandb.save('./outputs/train_labels.csv')

                    with torch.no_grad():
                        test_data = test_loader[0]
                        test_labels = torch.unsqueeze(test_loader[1],1)
    
                        out = model(test_data)
                        tloss = loss(out,test_labels)
                        tst_losses.append(tloss.item())

                        if epoch == (epochs-1):

                            # 1) Convert predictions of the test labels in the last epoch to a dataframe. (y_)
                            tst_out.append(out.view(1,-1).squeeze(0).tolist()) 
        
                            # unpack all tr_out from list of lists to a one list.
                            tst_out_list = list(itertools.chain.from_iterable(tst_out))
                            test_out_table =  pd.DataFrame(tst_out_list, columns = [str(epoch)])

                            # Save to csv and upload to wandb.
                            test_out_table.to_csv('./outputs/test_out.csv')
                            wandb.save('./outputs/test_out.csv')

                            # 2) Convert test labels to a dataframe. (y)
                            test_labels_flatt = test_labels.view(1,-1).squeeze(0).tolist()
                            test_labels_table =  pd.DataFrame(test_labels_flatt, columns = [str(epoch)])

                            test_labels_table.to_csv('./outputs/test_labels.csv')
                            wandb.save('./outputs/test_labels.csv')


                # save the train losses in the runs table future calculation.
                train_loss_table = pd.concat([train_loss_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save the train losses in the runs table future calculation.
                test_loss_table = pd.concat([test_loss_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.") 
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})


                # # check test loss for categorical y_pred
                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0].cuda(0)
                #         test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss)

            # Save the train and test losses to .csv files and upload it to the server.

            time = str(datetime.now())
            train_loss_table.to_csv('./outputs/train_loss_at'+time+'.csv')
            test_loss_table.to_csv('./outputs/test_loss_at'+time+'.csv')
            wandb.save('./outputs/train_loss_at'+time+'.csv')
            wandb.save('./outputs/test_loss_at'+time+'.csv')

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

            train_loss_table = pd.DataFrame()
            test_loss_table = pd.DataFrame()

            # Dataframes for saving the training and testing labels into .csv files.
            train_labels_table = pd.DataFrame()
            test_labels_table = pd.DataFrame()

            # Dataframes for saving the training and testing predictions into .csv files.
            train_out_table = pd.DataFrame()
            test_out_table = pd.DataFrame()

            for epoch in range(epochs):
                #Saving the train and test losses for logging and visualization purposes.
                tr_losses = []
                tst_losses = []

                # Saving the train and test predictions for logging and visualization purposes.
                tr_out = []
                tst_out = []

                # Saving the train and test labels for logging and visualization purposes.
                tr_labels = []
                tst_labels = []

                for i,(batch, labels, lbls_noises, noises_vars) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    batch = batch.cuda(0)
                    labels = torch.unsqueeze(labels,1).cuda(0)
                    noises_vars = torch.unsqueeze(noises_vars,1).type(torch.float32).cuda(0)
                    
                    out = model(batch)
                    mloss = loss(out,labels,noises_vars)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()

                    if epoch == (epochs-1):
                         # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                        tr_out.append(out.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_out_list = list(itertools.chain.from_iterable(tr_out))
                        train_out_table =  pd.DataFrame(tr_out_list, columns = [str(epoch)])

                        train_out_table.to_csv('/final_outps/train_out.csv') 
                        wandb.save('/final_outps/train_out.csv')

                         # 2) Convert the train labels to a dataframe. (y)
                        tr_labels.append(labels.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_labels_list = list(itertools.chain.from_iterable(tr_labels))
                        train_labels_table =  pd.DataFrame(tr_labels_list, columns = [str(epoch)])


                        # Save the train labels to csv file ,then upload it to wandb.
                        train_labels_table.to_csv('/final_outps/train_labels.csv')
                        wandb.save('/final_outps/train_labels.csv')


                    with torch.no_grad():

                        test_data = test_loader[0].cuda(0)
                        test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                        out = model(test_data)
                        tloss = mse_loss(out,test_labels)
                        tst_losses.append(tloss.item())

                        if epoch == (epochs-1):

                            # 1) Convert predictions of the test labels in the last epoch to a dataframe. (y_)
                            tst_out.append(out.view(1,-1).squeeze(0).tolist()) 
        
                            # unpack all tr_out from list of lists to a one list.
                            tst_out_list = list(itertools.chain.from_iterable(tst_out))
                            test_out_table =  pd.DataFrame(tst_out_list, columns = [str(epoch)])

                            # Save to csv and upload to wandb.
                            test_out_table.to_csv('/final_outps/test_out.csv')
                            wandb.save('/final_outps/test_out.csv')

                            # 2) Convert test labels to a dataframe. (y)
                            test_labels_flatt = test_labels.view(1,-1).squeeze(0).tolist()
                            test_labels_table =  pd.DataFrame(test_labels_flatt, columns = [str(epoch)])

                            test_labels_table.to_csv('/final_outps/test_labels.csv')
                            wandb.save('/final_outps/test_labels.csv')

                        

                # save the train losses in the runs table future calculation.
                train_loss_table = pd.concat([train_loss_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save the train losses in the runs table future calculation.
                test_loss_table = pd.concat([test_loss_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.") 
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})


                # # check test loss for categorical y_pred
                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0].cuda(0)
                #         test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss)

            # Save the train and test losses to .csv files and upload it to the server.

            time = str(datetime.now())
            train_loss_table.to_csv('/final_outps/train_loss_at'+time+'.csv')
            test_loss_table.to_csv('/final_outps/test_loss_at'+time+'.csv')
            wandb.save('/final_outps/train_loss_at'+time+'.csv')
            wandb.save('/final_outps/test_loss_at'+time+'.csv')

        else:
            #################################################
            print("Running without Cuda support")
            ##################################################
            # Create mse loss for the testing part.
            mse_loss = MSELoss()

            train_loss_table = pd.DataFrame()
            test_loss_table = pd.DataFrame()

            # Dataframes for saving the training and testing labels into .csv files.
            train_labels_table = pd.DataFrame()
            test_labels_table = pd.DataFrame()

            # Dataframes for saving the training and testing predictions into .csv files.
            train_out_table = pd.DataFrame()
            test_out_table = pd.DataFrame()
            
            for epoch in range(epochs):
                #Saving the train and test losses for logging and visualization purposes.
                tr_losses = []
                tst_losses = []

                # Saving the train and test predictions for logging and visualization purposes.
                tr_out = []
                tst_out = []

                # Saving the train and test labels for logging and visualization purposes.
                tr_labels = []
                tst_labels = []

                for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
                    optimizer.zero_grad() 

                    batch = batch
                    labels = torch.unsqueeze(labels,1)
                    noises_var = torch.unsqueeze(noises_var,1).type(torch.float32)

                    out = model(batch)
                    mloss = loss(out,labels,noises_var)
                
                    tr_losses.append(mloss.item())
                    mloss.backward()
            
                    optimizer.step()

                    if epoch == (epochs-1):
                         # 1) Convert predictions of the train labels in the last epoch to a dataframe. (y_)
                        tr_out.append(out.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_out_list = list(itertools.chain.from_iterable(tr_out))
                        train_out_table =  pd.DataFrame(tr_out_list, columns = [str(epoch)])

                        train_out_table.to_csv('./outputs/train_out.csv') 
                        wandb.save('./outputs/train_out.csv')

                         # 2) Convert the train labels to a dataframe. (y)
                        tr_labels.append(labels.view(1,-1).squeeze(0).tolist())
                      
                        # unpack all tr_out from list of lists to a one list.
                        tr_labels_list = list(itertools.chain.from_iterable(tr_labels))
                        train_labels_table =  pd.DataFrame(tr_labels_list, columns = [str(epoch)])


                        # Save the train labels to csv file ,then upload it to wandb.
                        train_labels_table.to_csv('./outputs/train_labels.csv')
                        wandb.save('./outputs/train_labels.csv')

                    with torch.no_grad():
                        test_data = test_loader[0]
                        test_labels = torch.unsqueeze(test_loader[1],1)
    
                        out = model(test_data)
                        tloss = mse_loss(out,test_labels)
                        tst_losses.append(tloss.item())

                        if epoch == (epochs-1):

                            # 1) Convert predictions of the test labels in the last epoch to a dataframe. (y_)
                            tst_out.append(out.view(1,-1).squeeze(0).tolist()) 
        
                            # unpack all tr_out from list of lists to a one list.
                            tst_out_list = list(itertools.chain.from_iterable(tst_out))
                            test_out_table =  pd.DataFrame(tst_out_list, columns = [str(epoch)])

                            # Save to csv and upload to wandb.
                            test_out_table.to_csv('./outputs/test_out.csv')
                            wandb.save('./outputs/test_out.csv')

                            # 2) Convert test labels to a dataframe. (y)
                            test_labels_flatt = test_labels.view(1,-1).squeeze(0).tolist()
                            test_labels_table =  pd.DataFrame(test_labels_flatt, columns = [str(epoch)])

                            test_labels_table.to_csv('./outputs/test_labels.csv')
                            wandb.save('./outputs/test_labels.csv')



                # save the train losses in the runs table future calculation.
                train_loss_table = pd.concat([train_loss_table, pd.DataFrame(tr_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)
                
                # save the train losses in the runs table future calculation.
                test_loss_table = pd.concat([test_loss_table, pd.DataFrame(tst_losses, columns = ['#epoch_'+str(epoch)])], axis = 1)

                print('Epoch:', epoch, "has finished.") 
                # log the results to wandb 
                for i in range(len(tr_losses)):
                    wandb.log({"train loss": tr_losses[i], "test loss":tst_losses[i]})


                # # check test loss for categorical y_pred
                # if epoch==19:
                #     with torch.no_grad():
                #         test_data = test_loader[0].cuda(0)
                #         test_labels = torch.unsqueeze(test_loader[1],1).cuda(0)

                #         y_red_groups = group_testing(test_data, test_labels, 10, model, loss)

            # Save the train and test losses to .csv files and upload it to the server.

            time = str(datetime.now())
            train_loss_table.to_csv('./outputs/train_loss_at'+time+'.csv')
            test_loss_table.to_csv('./outputs/test_loss_at'+time+'.csv')
            wandb.save('./outputs/train_loss_at'+time+'.csv')
            wandb.save('./outputs/test_loss_at'+time+'.csv')
            
