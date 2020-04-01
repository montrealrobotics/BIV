from datetime import datetime

import pandas as pd
import torch
from torch.nn import MSELoss

import wandb


class Trainer:

    def __init__(self):
        # Check Cuda avaliability
        self.cuda = torch.cuda.is_available()

    def train(self, train_loader, test_loader, model,loss, optimizer,epochs):
        if self.cuda:
            ############################
            print("Running with Cuda support")
            ##############################

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
                for i,(batch, labels, lbls_noise, noises_var) in enumerate(train_loader):
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
        if self.cuda:
            #################################################
            print("Running using Cuda support")
            ##################################################

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

        if self.cuda:
            #################################################
            print("Running using Cuda support")
            ##################################################

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

                    mloss = loss(out,labels_noisy,noises_var)
                
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

                    with torch.no_grad():
                        test_data = test_loader[0]
                        test_labels = torch.unsqueeze(test_loader[1],1)
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
            train_runs_table.to_csv('./outputs/train_iv_run_at'+time+'.csv')
            test_runs_table.to_csv('./outputs/test_iv_run_at'+time+'.csv')

            #  Save the files in .csv format.
            wandb.save('./outputs/train_iv_run_at'+time+'.csv')
            wandb.save('./outputs/test_iv_run_at'+time+'.csv')