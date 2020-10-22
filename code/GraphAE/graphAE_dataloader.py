import numpy as np
import os
from plyfile import PlyData, PlyElement
import torch
#from multiprocessing.dummy import Pool as ThreadPool 

import transforms3d.euler as euler
from os import mkdir
from os.path import join, exists
import h5py


SCALE = 1

def get_ply_fn_lst(folder_lst):
    ply_fn_lst =[]
    for folder in folder_lst:
        name_lst = os.listdir(folder)
        for name in name_lst:
            if(".ply" in name):
                fn=folder + "/"+name
                ply_fn_lst+=[fn]
    return ply_fn_lst


##return p_num*3 np
def get_pc_from_ply_fn(ply_fn):
    plydata = PlyData.read(ply_fn)
    x=np.array(plydata['vertex']['x'])
    y=np.array(plydata['vertex']['y'])
    z=np.array(plydata['vertex']['z'])
    
    
    pc = np.array([x,y,z])
    pc = pc.transpose()
    
    pc = pc- pc.mean(0)
    pc=pc*SCALE
    
    
    
    return pc



"""
def get_pc_from_ply_fn_normalized(ply_fn):
    pc = get_pc_from_ply_fn(ply_fn)
    
    #pc = (pc-pc_mean)/pc_std
    
    return pc
"""

#pc p_num*3 torch tensor
def get_augmented_pc(pc):
    size= pc.shape[0]

    scale = np.random.rand()*0.2+0.9 #scale 0.9 to 1.1

    new_pc = pc*scale

    #axis = np.array([0,1,0])
    axis = np.random.rand(3)
    axis = axis / np.sqrt(pow(axis,2).sum())
    #theta=(np.random.randint(0,30)-15)/180.0*np.pi
    theta= (np.random.rand()-0.5)*np.pi*2
    
    R = euler.axangle2mat(axis,theta).reshape((1,3,3)).repeat(size,0)
    #T = np.random.randn(1,3,1)/50
    T = (np.random.rand(1,3,1)-0.5)*0.2 #-10cm to 10cm
    #T[:,1] = T[:,1]*0
    T=T.repeat(size,0)

    R = torch.FloatTensor(R).cuda()
    T = torch.FloatTensor(T).cuda()
    new_pc = torch.matmul(R, new_pc.view(size,3,1))+T

    #new_pc = np.matmul(R, new_pc.reshape((size,3,1))) +T
    
    return new_pc.view(size,3)


#pc batch*p_num*3 torch tensor
def get_augmented_pcs(pcs):
    batch=pcs.shape[0]
    size= pcs.shape[1]

    scale = torch.rand(batch)*0.2+0.9 #scale 0.7 to 1.1
    scale = scale.cuda()  #batch
    #print (scale)
    T = (torch.rand(batch,3)-0.5)*0.2 #-10cm to 10cm
    T = T.cuda() #batch*3

    R = []
    for i in range(batch):
        axis = np.random.rand(3)
        axis = axis / np.sqrt(pow(axis,2).sum())
        theta= (np.random.rand()-0.5)*np.pi*2
        mat = euler.axangle2mat(axis,theta) #3,3
        R +=[mat]

    R=torch.FloatTensor(np.array(R)).cuda() #batch*3*3


    new_pcs  = torch.einsum('b,bsc->bsc',[scale, pcs])
    new_pcs  = torch.einsum('bdc,bsc->bsd',[R, new_pcs])
    new_pcs = new_pcs + T.view(batch,1,3).repeat(1,size,1)

    return new_pcs


##return batch*p_num*3 torch
def get_random_pc_batch_from_ply_fn_lst_torch(ply_fn_lst , batch, augmented=False):
    
    ply_fn_batch = []
    
    for b in range(batch):
        index = np.random.randint(0, len(ply_fn_lst))
        ply_fn_batch+=[ply_fn_lst[index]]
    
    #index = np.random.randint(0,len(ply_fn_lst)-batch)
    #ply_fn_batch =ply_fn_lst[index:index+batch]
    #pool = ThreadPool(min(batch,60)) 
    #pc_batch = pool.map(get_pc_from_ply_fn, ply_fn_batch)
    
    pc_batch=torch.FloatTensor([]).cuda()
    for ply_fn in ply_fn_batch:
        pc = get_pc_from_ply_fn(ply_fn)
        new_pc = torch.FloatTensor(pc).cuda()
        if(augmented==True):
            new_pc = get_augmented_pc(new_pc)
        pc_batch  = torch.cat((pc_batch,new_pc.unsqueeze(0)),0)
    
    return pc_batch

#num*p_num*3 numpy
def get_all_pcs_from_ply_fn_lst_np(ply_fn_lst):
    pc_lst=[]
    n=0
    for ply_fn in ply_fn_lst:
        pc = get_pc_from_ply_fn(ply_fn)
        pc_lst +=[pc]
        if(n%100==0):
            print (n)
        n=n+1
    print ("load", n, "pcs")
    
    return pc_lst


##return batch*p_num*3 torch   batch*p_num*3 torch
def get_random_pc_batch_from_pc_lst_torch(pc_lst , neighbor_lst, neighbor_num_lst, batch, augmented=False):
    
    pcs_index_lst = np.random.randint(0, len(pc_lst), batch)
    pcs = pc_lst[pcs_index_lst]
    pc_batch = torch.FloatTensor(pcs).cuda()

    if(augmented==True):
        pc_batch=get_augmented_pcs(pc_batch)
    
    return pc_batch


#point_num*3
def compute_and_save_ply_mean(folder_lst, pc_fn):
    ply_fn_lst=get_ply_fn_lst(folder_lst)
    pc_batch = []
    for ply_fn in ply_fn_lst:
        pc = get_pc_from_ply_fn(ply_fn)
        pc_batch +=[pc]
    pc_batch = np.array(pc_batch)
    pc_mean  = pc_batch.mean(0)
    pc_std = pc_batch.std(0)
    np.save(pc_fn+"mean", pc_mean)
    np.save(pc_fn+"std", pc_std)
    return pc_mean ,pc_std
    
    
    
    

#pc p_num np*3
#template_ply Plydata
def save_pc_into_ply(template_ply, pc, fn):
    plydata=template_ply
    #pc = pc.copy()*pc_std + pc_mean
    plydata['vertex']['x']=pc[:,0]
    plydata['vertex']['y']=pc[:,1]
    plydata['vertex']['z']=pc[:,2]
    plydata.write(fn)
    
#pc p_num np*3
#color p_num np*3 (0-255)
#template_ply Plydata
def save_pc_with_color_into_ply(template_ply, pc, color, fn):
    plydata=template_ply
    #pc = pc.copy()*pc_std + pc_mean
    plydata['vertex']['x']=pc[:,0]
    plydata['vertex']['y']=pc[:,1]
    plydata['vertex']['z']=pc[:,2]
    
    plydata['vertex']['red']=color[:,0]
    plydata['vertex']['green']=color[:,1]
    plydata['vertex']['blue']=color[:,2]
    
    plydata.write(fn)
    plydata['vertex']['red']=plydata['vertex']['red']*0+0.7*255
    plydata['vertex']['green']=plydata['vertex']['red']*0+0.7*255
    plydata['vertex']['blue']=plydata['vertex']['red']*0+0.7*255
    
    
def get_smoothed_pc_batch_iter(pc, neighbor_lst,neighbor_num_lst, iteration=10):
    smoothed_pc = get_smoothed_pc_batch(pc,neighbor_lst,neighbor_num_lst)
    for i in range(iteration):
        smoothed_pc = get_smoothed_pc_batch(smoothed_pc,neighbor_lst,neighbor_num_lst)
    return smoothed_pc

#pc batch*point_num*3
#neibhor_lst point_num*max_neighbor_num
#neibhor_num_lst point_num
def get_smoothed_pc_batch(pc, neighbor_lst, neighbor_num_lst):
    batch = pc.shape[0]
    point_num = pc.shape[1]
    pc_padded = np.concatenate((pc, np.zeros((batch, 1,3))),1) #batch*(point_num+1)*1
    smoothed_pc = pc.copy()
    for n in range(1,neighbor_lst.shape[1]):
        smoothed_pc += pc_padded[:,neighbor_lst[:,n]]
        
    smoothed_pc =smoothed_pc / neighbor_num_lst.reshape((1,point_num,1)).repeat(batch,0).repeat(3, 2)
    
    return smoothed_pc

def get_smoothed_pc_iter(pc, neighbor_lst,neighbor_num_lst, iteration=10):
    smoothed_pc = get_smoothed_pc(pc,neighbor_lst,neighbor_num_lst)
    for i in range(iteration):
        smoothed_pc = get_smoothed_pc(smoothed_pc,neighbor_lst,neighbor_num_lst)
    return smoothed_pc


#pc point_num*3
#neibhor_lst point_num*max_neighbor_num
#neibhor_num_lst point_num
def get_smoothed_pc(pc, neighbor_lst, neighbor_num_lst):
    point_num =pc.shape[0]
    pc_padded = np.concatenate((pc, np.zeros((1,3))),0) #batch*(point_num+1)*1
    smoothed_pc = pc.copy()
    for n in range(1,neighbor_lst.shape[1]):
        smoothed_pc += pc_padded[neighbor_lst[:,n]]

    smoothed_pc =smoothed_pc / neighbor_num_lst.reshape(point_num,1).repeat(3,1)
    
    return smoothed_pc


def transform_plys_to_npy(ply_folder, npy_fn):
    pcs = []
    name_lst = os.listdir(ply_folder)
    n=0
    for name in name_lst:
        if(".ply" in name):
            if(n%100==0):
                print (n)
            fn = ply_folder+"/"+name
            pc = get_pc_from_ply_fn(fn)
            pcs+=[pc]
            n+=1
            
    pcs = np.array(pcs)
    
    np.save(npy_fn, pcs)
    
    
def get_pcs_from_ply_folder(ply_folder):
    pcs = []
    name_lst = os.listdir(ply_folder)
    n=0
    for name in name_lst:
        if(".ply" in name):
            if(n%100==0):
                print (n)
            fn = ply_folder+"/"+name
            pc = get_pc_from_ply_fn(fn)
            pcs+=[pc]
            n+=1
            
    pcs = np.array(pcs)
    
    return pcs
    
#transform_plys_to_npy("/diskb/body/data/yaser_7k/Pass1_Meshes_train", "/diskb/body/data/yaser_7k/Pass1_Meshes_train/pcs")
































