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
import os

    
    
def evaluate(param, model,  pc_name, pc, mean, template_plydata, out_ply_folder):
    
    in_pc_batch = torch.FloatTensor(pc).unsqueeze(0).cuda()
    out_pc_batch = model(in_pc_batch)
    
    out_pc = out_pc_batch[0]
    
    diff_pc = torch.pow(torch.pow(pc-out_pc,2).sum(1),0.5) #point_num
    MSE = diff_pc.mean()
    print (MSE)
    
    out_pc = np.array(out_pc.data.tolist()) #*np.array([1,-1,1])
    in_pc = np.array(pc.data.tolist())#*np.array([1,-1,1])
    diff_pc = np.array(diff_pc.data.tolist())
    
    Dataloader.save_pc_into_ply(template_plydata, out_pc, out_ply_folder+pc_name+".ply")
    
    
    return MSE
    


##return p_num*3 np
def get_pc_from_ply_fn(ply_fn):
    plydata = PlyData.read(ply_fn)
    x=np.array(plydata['vertex']['x'])
    y=np.array(plydata['vertex']['y'])
    z=np.array(plydata['vertex']['z'])
    
    pc = np.array([x,y,z])
    pc = pc.transpose()
    mean = pc.mean(0)
    SCALE=1#0.001
    pc = pc- mean
    pc=pc*SCALE
    
    return pc, mean

def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_indices']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
        
    return faces
    

def test(param, test_ply_fn, start_id, end_id, out_ply_folder):
    
    
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param)
    
    model.cuda()
    
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    
    model.eval()
    
    template_plydata = PlyData.read(param.template_ply_fn)
    
    MSE_sum=0
    MSE_num=0
    for i in range(start_id, end_id):
        ply_fn = test_ply_fn+"%04d"%i+".ply"
        if(os.path.exists(ply_fn)==False):
            continue
        pc, mean = get_pc_from_ply_fn(ply_fn)
        
        MSE = evaluate(param, model,  "%04d"%i, pc, mean, template_plydata,  out_ply_folder)   
        MSE_sum=MSE_sum+MSE
        MSE_num= MSE_num+1
    print ("mean MSE:", MSE_sum/MSE_num)
    

param=Param.Parameters()
param.read_config("../../train/1216_GraphAE18_fixbug/00_res_6p_smaller_harder.config")

#param.read_weight_path = "../../train/0820_GraphAE12/weight_41/model_0288000.weight"
param.read_weight_path = "../../train/1216_GraphAE18_fixbug/weight_00/model_0120000.weight"

test_ply_fn = "/diskb/body/data/yaser_7k/test_with_donglai/input_"
start_id = 50
end_id  =151

out_ply_folder = "/diskb/body/train/0820_GraphAE12/test_21/out_ply_donglai/"

with torch.no_grad():
    test(param, test_ply_fn, start_id, end_id, out_ply_folder)


        
        
