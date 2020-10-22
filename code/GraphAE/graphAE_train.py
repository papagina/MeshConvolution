# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
import graphAE as graphAE
import graphAE_param as Param
import graphAE_dataloader as Dataloader
from datetime import datetime
from plyfile import PlyData

            

def train_one_iteration(param, model, optimizer,pc_lst, epoch, iteration):
    optimizer.zero_grad()
    #start=datetime.now()
    in_pc_batch = Dataloader.get_random_pc_batch_from_pc_lst_torch(pc_lst,param.neighbor_id_lstlst, param.neighbor_num_lst, param.batch, param.augmented_data) #batch*3*point_num
    
    #in_pc_batch =  in_pc_batch -pcs_mean_torch
    #start=datetime.now()
    out_pc_batch = model(in_pc_batch)
    
    #in_pc_batch=in_pc_batch+pcs_mean_torch
    #out_pc_batch = out_pc_batch+pcs_mean_torch
    #print ("model forward time" , datetime.now()-start)
    
    #start=datetime.now()
    loss_pose = torch.zeros(1).cuda()
    loss_laplace = torch.zeros(1).cuda()
    if(param.w_pose >0):
        loss_pose = model.compute_geometric_loss_l1(in_pc_batch, out_pc_batch)
    if(param.w_laplace >0):
        loss_laplace = model.compute_laplace_loss_l1(in_pc_batch, out_pc_batch)
    
    loss = loss_pose*param.w_pose +  loss_laplace * param.w_laplace
    loss.backward()

    #model.analyze_gradients()

    optimizer.step()
    
    #print ("model optimize time" , datetime.now()-start)
    total_iteration = epoch*param.iter_per_epoch + iteration
    if(iteration%100 == 0):
        print ("###Epoch", epoch, "Iteration", iteration, total_iteration)
        if(param.w_pose>0):
            print ("loss_pose:", loss_pose.item())
        if(param.w_laplace>0):
            print ("loss_laplace:", loss_laplace.item())
        print ("loss:", loss.item())
        print ("lr:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        
    if(iteration%10 == 0): 
        param.logger.add_scalars('Train_loss_without_weight', {'Loss_pose': loss_pose.item()},total_iteration)
        param.logger.add_scalars('Train_loss_with_weight', {'Loss_pose': (loss_pose*param.w_pose).item()},total_iteration)
        
def evaluate(param, model, pc_lst,epoch,template_plydata, suffix, log_eval=True):
    geo_error_sum = 0
    pc_num = len(pc_lst)
    n = 0

    while (n<(pc_num-1)):
        batch = min(pc_num-n, param.batch)
        pcs = pc_lst[n:n+batch]

        pcs_torch = torch.FloatTensor(pcs).cuda()
        if(param.augmented_data==True):
            pcs_torch = Dataloader.get_augmented_pcs(pcs_torch)
        if(batch<param.batch):
            pcs_torch = torch.cat((pcs_torch, torch.zeros(param.batch-batch, param.point_num, 3).cuda()),0)
        out_pcs_torch = model(pcs_torch)
        geo_error_sum = geo_error_sum + model.compute_geometric_mean_euclidean_dist_error(pcs_torch, out_pcs_torch)*batch

        if(n==0):   
            out_pc = np.array(out_pcs_torch[0].data.tolist()) 
            gt_pc = np.array(pcs_torch[0].data.tolist()) 
            Dataloader.save_pc_into_ply(template_plydata, out_pc, param.write_tmp_folder+"epoch%04d"%epoch+"_out_"+suffix+".ply")
            Dataloader.save_pc_into_ply(template_plydata, gt_pc, param.write_tmp_folder+"epoch%04d"%epoch+"_gt_"+suffix+".ply")
        n = n+batch

    geo_error_avg=geo_error_sum.item()/pc_num
    if(log_eval==True):
        param.logger.add_scalars('Evaluate', {'MSE Geo Error': geo_error_avg}, epoch)
        print ("MSE Geo Error:", geo_error_avg)
     
    return geo_error_avg

def test(param, model, pc_lst, epoch, log_eval=True):
    geo_error_sum = 0
    pc_num = len(pc_lst)
    n = 0

    while (n<(pc_num-1)):
        batch = min(pc_num-n, param.batch)
        pcs = pc_lst[n:n+batch]

        pcs_torch = torch.FloatTensor(pcs).cuda()
        if(param.augmented_data==True):
            pcs_torch = Dataloader.get_augmented_pcs(pcs_torch)
        
        out_pcs_torch = model(pcs_torch)
        geo_error_sum = geo_error_sum + model.compute_geometric_mean_euclidean_dist_error(pcs_torch, out_pcs_torch)*batch
        n = n+batch

    geo_error_avg=geo_error_sum.item()/pc_num
    if(log_eval==True):
        param.logger.add_scalars('Test', {'MSE Geo Error': geo_error_avg}, epoch)
        print ("MSE Geo Error:", geo_error_avg)
    return geo_error_avg
    

def train(param):
    torch.manual_seed(0)
    np.random.seed(0)
    
    
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param)
    
    model.cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr=param.lr, weight_decay=param.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, param.lr_decay_epoch_step,gamma=param.lr_decay)

    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.compute_param_num()
    
    model.train()
    
    
    
    print ("**********Get training ply fn list from**********", param.pcs_train)
    ##get ply file lst
    pc_lst_train = np.load(param.pcs_train)
    param.iter_per_epoch = int(len(pc_lst_train) / param.batch)
    param.end_iter = param.iter_per_epoch * param.epoch
    print ("**********Get evaluating ply fn list from**********", param.pcs_evaluate)
    pc_lst_evaluate = np.load(param.pcs_evaluate)
    #print ("**********Get test ply fn list from**********", param.pcs_evaluate)
    #pc_lst_test = np.load(param.pcs_test)

    np.random.shuffle(pc_lst_train)
    np.random.shuffle(pc_lst_evaluate)


    pc_lst_train[:,:,0:3] -= pc_lst_train[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)
    pc_lst_evaluate[:,:,0:3] -= pc_lst_evaluate[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)
    #pc_lst_test[:,:,0:3] -= pc_lst_test[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)

    template_plydata = PlyData.read(param.template_ply_fn)
    
    print ("**********Start Training**********")
    
    min_geo_error=123456
    for i in range(param.start_epoch, param.epoch+1):

        if(((i%param.evaluate_epoch==0)and(i!=10)) or(i==param.epoch)):
            print ("###Evaluate", "epoch", i, "##########################")
            with torch.no_grad():
                torch.manual_seed(0)
                np.random.seed(0)
                geo_error = evaluate(param, model, pc_lst_evaluate,i,template_plydata, suffix="_eval")    
                if(geo_error<min_geo_error):
                    min_geo_error=geo_error
                    print ("###Save Weight")
                    path = param.write_weight_folder + "model_epoch%04d"%i +".weight"
                    torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, path)
                
            
        torch.manual_seed(i)
        np.random.seed(i)

        for j in range(param.iter_per_epoch):
            train_one_iteration(param, model, optimizer,pc_lst_train, i, j)
        
        scheduler.step()

        

param=Param.Parameters()
param.read_config("../../train/0422_graphAE_dfaust/30_conv_res.config")

train(param)


        
        
