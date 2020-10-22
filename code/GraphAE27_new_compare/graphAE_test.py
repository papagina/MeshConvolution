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
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from renderer.renderer import Mesh, Renderer
from matplotlib import cm

  

def evaluate(param, model, pc_lst):
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
        geo_error = model.compute_geometric_mean_euclidean_dist_error(pcs_torch, out_pcs_torch)
        geo_error_sum = geo_error_sum + geo_error*batch
        print(n, geo_error)

        n = n+batch
        

    geo_error_avg=geo_error_sum.item()/pc_num
    
     
    return geo_error_avg


def test(param,test_npy_fn, out_ply_folder, out_img_folder, is_render_mesh=False, skip_frames =0):
    
    
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param)
    
    model.cuda()
    
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.init_test_mode()
    
    
    model.train()
    
    
    print ("**********Get test pcs**********", test_npy_fn)
    ##get ply file lst
    pc_lst= np.load(test_npy_fn)

    print (pc_lst.shape[0], "meshes in total.")
    pc_lst[:,:,0:3] -= pc_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)

    geo_error_avg=evaluate(param, model, pc_lst)

    print ("geo error:", geo_error_avg)#, "laplace error:", laplace_error_avg)
    
        
    
def visualize_weights(param, out_folder):
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param, test_mode=False)
    
    model.cuda()
    
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    
    model.eval()
    
    model.quantify_and_draw_w_weight_histogram(out_folder)
    
    #draw_w_weight_histogram(model, out_folder )

    
    

param=Param.Parameters()
param.read_config("../../train/0223_GraphAE27_compare/40_conv_pool_FeaST.config")
print (param.use_vanilla_pool)
#param.augmented_data=True
param.batch =16

param.read_weight_path = "../../train/0223_GraphAE27_compare/weight_40/model_epoch0200.weight"
print (param.read_weight_path)

test_npy_fn = "../../data/D-FAUST_harder/test.npy"

out_test_folder = "../../train/0223_GraphAE27_compare/test_40/epoch200/"

out_ply_folder = out_test_folder+"ply/"
out_img_folder = out_test_folder+"img/"

out_weight_visualize_folder=out_test_folder+"weight_vis/"

out_interpolation_folder = out_test_folder+"interpolation/"

if not os.path.exists(out_ply_folder):
    os.makedirs(out_ply_folder)
    
if not os.path.exists(out_img_folder+"/gt"):
    os.makedirs(out_img_folder+"gt/")
    
if not os.path.exists(out_img_folder+"/out"):
    os.makedirs(out_img_folder+"out/")
    
if not os.path.exists(out_img_folder+"/out_color"):
    os.makedirs(out_img_folder+"out_color/")
    
if not os.path.exists(out_weight_visualize_folder):
    os.makedirs(out_weight_visualize_folder)

pc_lst= np.load(test_npy_fn)
print (pc_lst[:,:,1].max()-pc_lst[:,:,1].min())

with torch.no_grad():
    torch.manual_seed(2)
    np.random.seed(2)
    test(param, test_npy_fn, out_ply_folder,out_img_folder, is_render_mesh=False, skip_frames=0)
    test(param, param.pcs_train, out_ply_folder,out_img_folder, is_render_mesh=False, skip_frames=0)
    visualize_weights(param, out_weight_visualize_folder)
    #test_interpolation(param, middle_layer_id=7, inter_num=10, test_npy_fn=test_npy_fn, id1=40, id2=2000, out_folder=out_interpolation_folder)


        
        
