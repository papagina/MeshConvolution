import trimesh

import graphAE_dataloader as loader

import numpy as np
from plyfile import PlyData


#pc p_num np*3
#template_ply Plydata
def save_pc_into_ply(template_ply, pc, fn):
    plydata=template_ply
    #pc = pc.copy()*pc_std + pc_mean
    plydata['vertex']['x']=pc[:,0]
    plydata['vertex']['y']=pc[:,1]
    plydata['vertex']['z']=pc[:,2]
    plydata.write(fn)

i=0
template_plydata = PlyData.read('/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/fall/data/D-FAUST/sample.ply')
train = np.load('/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/fall/data/D-FAUST/test.npy')
print(template_plydata['face']['vertex_indices'])
out_pc = train[i]

save_pc_into_ply(template_plydata, out_pc, str(i)+".ply")



"""
def save_colored_mesh(vertex, face, filename):

	mesh = trimesh.Trimesh(vertices=vertex,
                       faces=face)

	#mesh.visual.vertex_colors = color
	mesh.export(filename)


i=1

sample_obj = trimesh.load_mesh('/mnt/hdd1/yi_hdd1/GraphCNN_Facebook/body/fall/data/D-FAUST/sample.ply')
faces = sample_obj.faces
vertices = train[i,:,:]
#vertices = sample.vertices

save_colored_mesh(vertices,faces , str(i)+ '.ply')

print(str(i) + '/' + str(train))

"""