import numpy as np
import os
from plyfile import PlyData, PlyElement
import graphAE_dataloader as Dataloader
import matplotlib as mpl
from matplotlib import cm


## point_num*max_neighbor_faces_num
## for vacant neighboring faces, the id is faces_num
def get_point_faces_lst_np(face_lst, point_num):
    point_faces_lst = [[]]*point_num
    
    for i in range(face_lst.shape[0]):
        p1, p2, p3 = face_lst[i]
        point_faces_lst[p1]=point_faces_lst[p1]+[i]
        point_faces_lst[p2]=point_faces_lst[p2]+[i]
        point_faces_lst[p3]=point_faces_lst[p3]+[i]
    
    
    faces_num = face_lst.shape[0]
    max_neighbor_faces_num=0
    for neighbors in point_faces_lst:
        num = len(neighbors)
        if(num>max_neighbor_faces_num):
            max_neighbor_faces_num=num
        
    point_faces_np = np.ones((point_num, max_neighbor_faces_num))*faces_num
    point_faces_np= point_faces_np.astype(int)
    
    for i in range(point_num):
        neighbors = np.array(point_faces_lst[i])
        num = neighbors.shape[0]
        point_faces_np[i,0:num] = neighbors
    
    return point_faces_np


#pcs batch*point_num*3
#face_lst faces_num*3 
#point_faces_id_lst point_num*max_neighbor_num
#pcs_area batch*point_nun
def get_pcs_area_np(face_lst, point_faces_np, pcs):
    
    p0s = pcs[:,face_lst[:,0]] #batch*faces_num*3
    p1s = pcs[:,face_lst[:,1]] #batch*faces_num*3
    p2s = pcs[:,face_lst[:,2]] #batch*faces_num*3
    
    cross = np.cross(p1s-p0s, p2s-p0s) #batch*faces_num*3
    
    faces_num = face_lst.shape[0]
    batch=pcs.shape[0]
    faces_areas = np.zeros((batch, faces_num+1))
    faces_areas[:,0:faces_num] = np.power(np.power(cross,2).sum(2),0.5)/2 #batch*faces_num
    
    
    point_faces_np=point_faces_np.astype(int)
    pcs_areas = faces_areas[:, point_faces_np] #batch*point_num*max_neighbor_num
    print (pcs_areas.shape)
    pcs_area = pcs_areas.sum(2)/3
    
    return pcs_area
    
    
    

    

pcs_fn = "../../data/D-FAUST_harder/train.npy"
pcs = np.load(pcs_fn)

out_weighted_pc_fn = "../../data/D-FAUST_harder/train_with_weights.npy"

template_ply_fn = "../../data/D-FAUST/sample.ply"


plydata = PlyData.read(template_ply_fn)
face_lst = np.array(plydata['face'].data['vertex_indices'].tolist()).astype(int)

point_num  = np.array(plydata['vertex']['x']).shape[0]
if((point_num -1)!=face_lst.max()):
    print ("point num doesnt match")
face_num = face_lst.shape[0]

print ("The mesh has", face_num,"faces and", point_num,"points")
print ("get point faces list")
point_faces_np = get_point_faces_lst_np(face_lst, point_num)
print (point_faces_np.shape)
print ("get points' area")
pcs_area = get_pcs_area_np(face_lst,point_faces_np, pcs)
area_max=pcs_area.max()
area_min=pcs_area.min()
print(area_max, area_min, pcs_area.mean())

pcs_area = pcs_area/pcs_area.sum(1).reshape((-1,1)).repeat(point_num,1)

print (pcs_area.shape, pcs_area.sum())

pcs_area = pcs_area.reshape((pcs.shape[0],pcs.shape[1],1))
pcs_with_weights = np.concatenate((pcs, pcs_area), 2) #batch*point_num*4
pcs_with_weights=pcs_with_weights.astype(np.float32)
np.save(out_weighted_pc_fn, pcs_with_weights)


###visualize weights
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    color = mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    (r,g,b) = mpl.colors.ColorConverter.to_rgb(color)
    
    return np.array([r,g,b])

def get_colors_from_diff_pc(diff_pc, min_error, max_error):
    colors = np.zeros((diff_pc.shape[0],3))
    #print (diff_pc.shape,min_error, max_error)
    cmap=cm.get_cmap('Spectral')
    for i in range(diff_pc.shape[0]):
        diff = diff_pc[i]
        #print(diff)
        mix = (diff-min_error)/(max_error-min_error)
        if(mix<0):
            mix=0
        if(mix>1):
            mix=1
        #color = colorFader('green','red', mix)
        #print (cmap(mix))
        #print (mix)
        r,g,b,a =cmap(mix)
        colors[i]=[r,g,b]
    return colors

pc = pcs[0,:]
weights=pcs_area[0,:,0]
colors=get_colors_from_diff_pc(weights, weights.min(), weights.max())*255

Dataloader.save_pc_with_color_into_ply(plydata, pc, colors, out_weighted_pc_fn +"_visualize_area_weights.ply")















