import numpy as np
import os
from plyfile import PlyData, PlyElement
import torch
from multiprocessing.dummy import Pool as ThreadPool 


def get_ply_fn_lst(folder_lst):
    ply_fn_lst =[]
    for folder in folder_lst:
        name_lst = os.listdir(folder)
        for name in name_lst:
            if(".ply" in name):
                fn=folder + "/"+name
                ply_fn_lst+=[fn]
    return ply_fn_lst


##return 3*p_num np
def get_pc_from_ply_fn(ply_fn):
    plydata = PlyData.read(ply_fn)
    x=np.array(plydata['vertex']['x'])
    y=np.array(plydata['vertex']['y'])
    z=np.array(plydata['vertex']['z'])
    
    pc = np.array([x,y,z])/100.0
    
    return pc


##return batch*3*p_num torch
def get_random_pc_batch_from_ply_fn_lst_torch(ply_fn_lst , batch):
    
    ply_fn_batch = []
    
    for b in range(batch):
        index = np.random.randint(0, len(ply_fn_lst))
        ply_fn_batch+=[ply_fn_lst[index]]
    
    #index = np.random.randint(0,len(ply_fn_lst)-batch)
    #ply_fn_batch =ply_fn_lst[index:index+batch]
    #pool = ThreadPool(min(batch,60)) 
    #pc_batch = pool.map(get_pc_from_ply_fn, ply_fn_batch)
    
    pc_batch=[]
    for ply_fn in ply_fn_batch:
        pc = get_pc_from_ply_fn(ply_fn)
        pc_batch +=[pc]
    
    pc_batch_torch = torch.FloatTensor(pc_batch).cuda()
    
    return pc_batch_torch


    
    
                
        
        






































