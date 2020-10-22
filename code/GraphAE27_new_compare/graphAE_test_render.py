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
from renderer.renderer import Mesh, Renderer
import cv2
import matplotlib as mpl
import os

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    color = mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    (r,g,b) = mpl.colors.ColorConverter.to_rgb(color)
    return np.array([r,g,b])

def get_colors_from_diff_pc(diff_pc, faces, min_error, max_error):
    colors = np.zeros((diff_pc.shape[0],3))
    for i in range(diff_pc.shape[0]):
        diff = diff_pc[i]
        mix = (diff-min_error)/(max_error-min_error)
        if(mix<0):
            mix=0
        if(mix>1):
            mix=1
        color = colorFader('green','red', mix)
        colors[i]=color
    colors_faces = colors[faces]
    colors_faces = colors_faces.mean(1)
    return colors_faces
    
    

def evaluate(param, model, render, pc_name, pc, mean, template_plydata, faces, out_ply_folder,out_render_gt_folder, out_render_out_folder):
    
    in_pc_batch = torch.FloatTensor(pc).unsqueeze(0).cuda()
    out_pc_batch = model(in_pc_batch)
    
    out_pc = out_pc_batch[0]
    
    diff_pc = torch.pow(torch.pow(pc-out_pc,2).sum(1),0.5) #point_num
    MSE = diff_pc.mean()
    print (MSE)
    
    out_pc = np.array(out_pc.data.tolist()) #*np.array([1,-1,1])
    in_pc = np.array(pc.data.tolist())#*np.array([1,-1,1])
    diff_pc = np.array(diff_pc.data.tolist())
    
    #Dataloader.save_pc_into_ply(template_plydata, out_pc, out_ply_folder+pc_name+".ply")
    
    camera_pos, camera_direction, camera_fov = np.array([2, 0, 0]), np.array([-1,0,0]), 45 
    
    out_pc_Mesh = Mesh(out_pc, faces, [], np.array([0.7,0.7,0.7]))
    out_pc_img = render.render_one_frame([out_pc_Mesh],  camera_pos, camera_direction, camera_fov)
    out_pc_img = (out_pc_img * 255).astype(np.uint8)
    cv2.imwrite(out_render_out_folder+pc_name+".png", cv2.cvtColor(out_pc_img, cv2.COLOR_RGB2BGR))
    
    
    in_pc_Mesh = Mesh(in_pc, faces, [], np.array([0.7,0.7,0.7]))
    #print (diff_pc.max())
    in_pc_Mesh.colors = get_colors_from_diff_pc(diff_pc,faces,0,0.05)
    in_pc_img = render.render_one_frame([in_pc_Mesh],  camera_pos, camera_direction, camera_fov)
    in_pc_img = (in_pc_img * 255).astype(np.uint8)
    cv2.imwrite(out_render_gt_folder+pc_name+".png", cv2.cvtColor(in_pc_img, cv2.COLOR_RGB2BGR))
    
    
    
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
    SCALE=0.001
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
    

def test(param, test_ply_fn, start_id, end_id, out_ply_folder,out_render_gt_folder, out_render_out_folder):
    
    
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param)
    
    model.cuda()
    
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    
    model.eval()
    
    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)
    
    from renderer.render.gl.glcontext import create_opengl_context
    from renderer.render.gl.glcontext import destroy_opengl_context
    RESOLUTION = 1024
    
    glutWindow=create_opengl_context(width=RESOLUTION, height=RESOLUTION)
    
    render = Renderer(width=RESOLUTION, height=RESOLUTION)
    MSE_sum=0
    MSE_num=0
    for i in range(start_id, end_id):
        ply_fn = test_ply_fn+"%06d"%i+".ply"
        if(os.path.exists(ply_fn)==False):
            continue
        pc, mean = get_pc_from_ply_fn(ply_fn)
        
        MSE = evaluate(param, model, render, "%06d"%i, pc, mean, template_plydata, faces, out_ply_folder,out_render_gt_folder, out_render_out_folder)   
        MSE_sum=MSE_sum+MSE
        MSE_num= MSE_num+1
    print ("mean MSE:", MSE_sum/MSE_num)
    destroy_opengl_context(glutWindow)
    

param=Param.Parameters()
param.read_config("../../train/0820_GraphAE12/41_res_mbias_6p.config")

param.read_weight_path = "../../train/0820_GraphAE12/weight_41/model_0288000.weight"

test_ply_fn = "/diskb/body/data/yaser_7k/all_ply/tracked-"
start_id = 3186
end_id  =5400

out_ply_folder = "/diskb/body/train/0820_GraphAE12/test_41/out_ply/"
out_render_gt_folder = "/diskb/body/train/0820_GraphAE12/test_41/render_gt_front/"
out_render_out_folder = "/diskb/body/train/0820_GraphAE12/test_41/render_out_front/"

with torch.no_grad():
    test(param, test_ply_fn, start_id, end_id, out_ply_folder,out_render_gt_folder, out_render_out_folder)


        
        
